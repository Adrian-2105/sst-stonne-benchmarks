import os, sys, glob
import argparse
import json
import subprocess
import re
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SstStonneBenchmarkUtils import LayerDataFilenames

"""
Given the root folder of a DNN model which contains the results of each of its layers
(results generated in the same or in a different folder using the LaunchSimulation.py script),
parses all the results to obtain a general result for the whole model, generating a 
"model_statistics.csv" file containing the results of the model.

If multiple results are found for a layer in the same folder, only the last one will be used.

Requirements:
  - (optional) pip install tabulate
    If satisfied, the script will generate a table in the terminal with the results of the model.
"""

OUTPUT_FILENAME = "model_statistics.csv"

class DataflowResult:
    """
    Class to read and store the results of a dataflow result for a layer
    """

    def __init__(self, layer_dir : str, dataflow : str):
        df_dir = os.path.join(layer_dir, dataflow)
        if not os.path.isdir(df_dir):
            print(f"[WARNING] {dataflow} result not found for layer {layer_dir} ({df_dir}), skipping...")
            raise Exception

        # check if all the result files are there
        try:
            cycles_file = glob.glob(os.path.join(df_dir, 'output_stats_*.counters'))[-1]
            stonne_stats_file = glob.glob(os.path.join(df_dir, 'output_stats_*.txt'))[-1]
            sst_stats_file = glob.glob(os.path.join(df_dir, 'output.csv'))[-1]
        except IndexError:
            print(f"[WARNING] {dataflow} result have missing files for layer {layer_dir}, skipping...")
            raise Exception

        # get cycles
        self.cycles = int(subprocess.check_output(f"cat {cycles_file} | grep CYCLES | head -n1 | cut -d'=' -f2", shell=True))

        # cache B hits
        # l1cache.latency_GetS_hit : Accumulator : SimTime = 33162522000; Rank = 0; Sum.u64 = 513104086; SumSQ.u64 = 16246995092; Count.u64 = 55699418; Min.u64 = 2; Max.u64 = 119;
        self.cache_B_hits = int(subprocess.check_output(f"cat {sst_stats_file} | grep latency_GetS_hit | tr ';' '\n' | grep Sum.u64 | tr -d ' ' | cut -d '=' -f2", shell=True))
        # cache B misses
        # l1cache.latency_GetS_miss : Accumulator : SimTime = 33162522000; Rank = 0; Sum.u64 = 20527632; SumSQ.u64 = 1774366648; Count.u64 = 237507; Min.u64 = 86; Max.u64 = 112; 
        self.cache_B_misses = int(subprocess.check_output(f"cat {sst_stats_file} | grep latency_GetS_miss | tr ';' '\n' | grep Sum.u64 | tr -d ' ' | cut -d '=' -f2", shell=True))
        # cache B total accesses (hits + misses)
        self.cache_B_total_accesses = self.cache_B_hits + self.cache_B_misses
        # cache B hit rate (hits / total accesses)
        self.cache_B_hit_rate = self.cache_B_hits / self.cache_B_total_accesses

        # stonne memory stats
        with open(stonne_stats_file, 'r') as stonne_stats:
            stonne_stats = json.load(stonne_stats)
            self.N_cycles_multiplying = stonne_stats['SDMemoryStats']['N_cycles_multiplying']
            self.N_cycles_merging = stonne_stats['SDMemoryStats']['N_cycles_merging']
            self.N_SRAM_weight_reads = stonne_stats['SDMemoryStats']['N_SRAM_weight_reads']
            self.N_SRAM_input_reads = stonne_stats['SDMemoryStats']['N_SRAM_input_reads']
            self.N_SRAM_psum_reads = stonne_stats['SDMemoryStats']['N_SRAM_psum_reads']
            self.N_SRAM_psum_writes = stonne_stats['SDMemoryStats']['N_SRAM_psum_writes']
            self.N_DRAM_psum_writes = stonne_stats['SDMemoryStats']['N_DRAM_psum_writes']
            self.N_multiplications = sum([ms['N_multiplications'] for ms in stonne_stats[
                'MSNetworkStats' if dataflow.startswith('inner') else 'SparseFlex_MSNetworkStats']['MSwitchStats']])
            

class LayerResults:
    """
    Class to read and store all the results of a layer (including all the dataflows)
    """

    def __init__(self, layer_dir):
        def extract_layer_number(layer_name):
            match = re.search(r'(\d+)$', layer_name)
            return int(match.group(1)) if match else -1

        # find layer information in the results to obtain the necessary metadata
        FILENAMES = LayerDataFilenames.get_layer_data_filenames()
        layer_info_file = glob.glob(os.path.join(layer_dir, '*', FILENAMES['layer_info']))
        if not layer_info_file:
            print(f"[ERROR] layer_info.json not found in any of the subfolders of {layer_dir}, aborting...")
            raise Exception

        # extract layer information
        with open(layer_info_file[-1], 'r') as layer_info:
            layer_info = json.load(layer_info)
            self.layer_name = layer_info['name']
            self.layer_number = extract_layer_number(layer_info['name'])
            self.M = layer_info['M']
            self.N = layer_info['N']
            self.K = layer_info['K']
            self.A_nnz = layer_info['A_bytes'] // 4
            self.B_nnz = layer_info['B_bytes'] // 4
            self.A_sparsity = 1 - self.A_nnz / (self.M * self.K)
            self.B_sparsity = 1 - self.B_nnz / (self.N * self.K)

        # get the results of each dataflow
        self.best_cycles = float('inf')
        self.best_dataflow = None
        for dataflow in ['inner_product_m', 'inner_product_n', 'outer_product_m', 'outer_product_n', 'gustavsons_m', 'gustavsons_n']:
            try:
                # extract the results from the dataflow
                dataflow_results = DataflowResult(layer_dir, dataflow)
                # create a dynamic field with the results of the dataflow
                setattr(self, dataflow, dataflow_results)

                if dataflow_results.cycles < self.best_cycles:
                    self.best_cycles = dataflow_results.cycles
                    self.best_dataflow = dataflow
            except:
                setattr(self, dataflow, None)
            

class LayerMapping:
    """
    Class used to store the mapping of a layer, given by compute_best_mappnig
    """
    
    DATAFLOWS = ['inner_product_m', 'inner_product_n', 'outer_product_m', 'outer_product_n', 'gustavsons_m', 'gustavsons_n']

    def __init__(self, layer_num, dataflow, layer_cycles, accum_cycles, EC, EC_cycles=0):
        self.layer_num = layer_num
        self.dataflow = dataflow
        self.dataflow_name = LayerMapping.DATAFLOWS[dataflow]
        self.layer_cycles = layer_cycles
        self.accum_cycles = accum_cycles
        self.needs_EC = EC
        self.EC_cycles = EC_cycles if EC else 0

    def __str__(self):
        return f"Layer {self.layer_num} - {self.dataflow_name} - {self.accum_cycles} cycles"


def find_best_mapping_path(layers : List[LayerResults], ec_cycles_elem : int) -> List[LayerMapping]:
    """
    Computes the best mapping path considering the explicit conversion costs

    layers: list of LayerResults
    ec_cycles_elem: cycles per element for explicit conversions

    returns: list of LayerMapping
    """

    # extract the cycles and build a table to be used in the DP algorithm
    layer_cycles = []
    for layer in layers:
        layer_cycles.append([layer.inner_product_m.cycles, 
                            layer.inner_product_n.cycles, 
                            layer.outer_product_m.cycles,
                            layer.outer_product_n.cycles,
                            layer.gustavsons_m.cycles,
                            layer.gustavsons_n.cycles])
        
    # compute the explicit conversion costs of each layer
    EC_CYCLES = [layer.A_nnz * ec_cycles_elem for layer in layers]

    """
    Transformations table (requires A matrix, returns C matrix):
    # inner_product_m -> requires CSR, returns CSR
    # inner_product_n -> requires CSR, returns CSC
    # outer_product_m -> requires CSC, return CSR
    # outer_product_n -> requires CSC, return CSC
    # gustavsons_m    -> requires CSR, returns CSR
    # gustavsons_n    -> requires CSC, returns CSC
    """
    TRANSITION_REQUIRES = [True, True, False, False, True, False]
    TRANSITION_RETURNS = [True, False, True, False, True, False]


    # the minimum cycles of the first layer are just 
    min_cycles = [layer_cycles[0]]

    # computes the minimum path (in cycles) to reach each layer
    for cur_layer in range(1, len(layer_cycles)):
        prev_layer = cur_layer - 1

        # init the current layer minimum cycles found
        min_cycles.append([float('inf') for _ in range(6)])

        # test each dataflow of the cur_layer against all the dataflows of the prev_layer
        for cur_df in range(len(LayerMapping.DATAFLOWS)):
            for prev_df in range(len(LayerMapping.DATAFLOWS)):
                # computes the cycles of having taken this path, adding EC costs if needed
                cycles = layer_cycles[cur_layer][cur_df] + min_cycles[prev_layer][prev_df]
                if TRANSITION_RETURNS[prev_df] != TRANSITION_REQUIRES[cur_df]:
                    cycles += EC_CYCLES[cur_layer]

                # update the minimum cycles found
                min_cycles[cur_layer][cur_df] = min(min_cycles[cur_layer][cur_df], cycles)


    # build the solution from the DP results, starting from the minimum accumulated cycles
    solution = []
    cur_df = min_cycles[-1].index(min(min_cycles[-1]))
    for cur_layer in range(len(min_cycles) - 1, 0, -1):
        prev_layer = cur_layer - 1

        # check the path taken to reach this layer
        for prev_df in range(len(LayerMapping.DATAFLOWS)):
            # computes the cycles of having taken this path, adding EC costs if needed
            cycles = min_cycles[prev_layer][prev_df] + layer_cycles[cur_layer][cur_df]
            if TRANSITION_RETURNS[prev_df] != TRANSITION_REQUIRES[cur_df]:
                cycles += EC_CYCLES[cur_layer]

            # if the path taken matches the current dataflow accumulated cycles, add the solution and move to the previous layer
            if cycles == min_cycles[cur_layer][cur_df]:
                solution.append(LayerMapping(cur_layer, cur_df, 
                                                layer_cycles[cur_layer][cur_df], min_cycles[cur_layer][cur_df], 
                                                TRANSITION_RETURNS[prev_df] != TRANSITION_REQUIRES[cur_df], EC_CYCLES[cur_layer]))
                cur_df = prev_df
                break
        else:
            raise Exception("There is an error in the DP solution or in the solution building")

    # add the first layer
    solution.append(LayerMapping(0, cur_df, layer_cycles[0][cur_df], min_cycles[0][cur_df], False))
    
    # reverse the solution to have the first layer first
    return solution[::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Obtain the results of a DNN model')
    parser.add_argument('results_model_dir', type=str, help='Directory containing the results of the model')
    parser.add_argument('-c', '--ec_cycles', type=int, default=20, help='Number of cycles of the Explicit Conversions per NNZ element (default: 20)')
    parser.add_argument('-o', '--output_filename', type=str, default=OUTPUT_FILENAME, help=f'Output filename (default: {OUTPUT_FILENAME})')
    args = parser.parse_args()

    # check if dir exists
    assert os.path.isdir(args.results_model_dir), f"Directory {args.results_model_dir} does not exist"


    # read all layers getting its individual results
    layers = []
    for layer_dir in os.listdir(args.results_model_dir):
        # check if it is a directory
        if not os.path.isdir(os.path.join(args.results_model_dir, layer_dir)):
            continue

        # get the layer results
        layers.append(LayerResults(os.path.join(args.results_model_dir, layer_dir)))
    layers = sorted(layers, key=lambda layer: layer.layer_number)
    

    # build best-mapping-path considering explicit conversions
    mapping_with_EC = find_best_mapping_path(layers, args.ec_cycles)
    # build best-mapping-path without considering explicit conversions
    mapping_without_EC = find_best_mapping_path(layers, float('inf'))


    # print a table with the results of the mapping
    try:
        import tabulate

        # print the results table for the mapping with explicit conversions
        print("MAPPINGS WITH EXPLICIT CONVERSIONS")
        print(tabulate.tabulate(
            [[f"Layer {i}", mapping_with_EC[i].dataflow_name, mapping_with_EC[i].layer_cycles, mapping_with_EC[i].accum_cycles, mapping_with_EC[i].needs_EC, mapping_with_EC[i].EC_cycles if mapping_with_EC[i].EC_cycles > 0 else None] for i in range(len(layers))],
            headers=["Layer", "Dataflow", "Layer Cycles", "Accum Cycles", "Needs EC?", "EC Cycles"],
            tablefmt="orgtbl"
        ))

        # print the results table for the mapping without explicit conversions
        print()
        print("MAPPINGS WITHOUT EXPLICIT CONVERSIONS")
        print(tabulate.tabulate(
            [[f"Layer {i}", mapping_without_EC[i].dataflow_name, mapping_without_EC[i].layer_cycles, mapping_without_EC[i].accum_cycles, mapping_without_EC[i].needs_EC] for i in range(len(layers))],
            headers=["Layer", "Dataflow", "Layer Cycles", "Accum Cycles", "Needs EC?"],
            tablefmt="orgtbl"
        ))
        print()
    except ImportError:
        print("tabulate package not found, install it with 'pip install tabulate' to show the results in a table")


    # generate the CSV with the results
    OUTPUT_FILE_FULLPATH = os.path.join(args.results_model_dir, args.output_filename)
    with open(OUTPUT_FILE_FULLPATH, 'w') as f:
        # create the header
        print(';'.join([
            # general information
            'layer_name',
            'M',
            'N',
            'K',
            'A_nnz',
            'B_nnz',
            'A_sparsity',
            'B_sparsity',

            # cycles of each dataflow
            'inner_product_m',
            'inner_product_n',
            'outer_product_m',
            'outer_product_n',
            'gustavsons_m',
            'gustavsons_n',
            'min_cycles',
            'best_dataflow',

            # EC
            'EC_cost_per_elem',

            # best dataflows mapping path considering EC
            'best_dataflow_with_EC',
            'needs_EC_from_prev_layer',
            'EC_cycles',
            'dataflow_cycles_with_EC',
            'total_layer_cycles_with_EC',
            'accum_cycles_with_EC',

            # best dataflows mapping path avoiding EC
            'best_dataflow_without_EC',
            'dataflow_cycles_without_EC',
            'accum_cycles_without_EC',

            # cache statistics
            'cache_B_hits_inner_product_m',
            'cache_B_misses_inner_product_m',
            'cache_B_total_accesses_inner_product_m',
            'cache_B_hit_rate_inner_product_m',
            'cache_B_hits_inner_product_n',
            'cache_B_misses_inner_product_n',
            'cache_B_total_accesses_inner_product_n',
            'cache_B_hit_rate_inner_product_n',
            'cache_B_hits_outer_product_m',
            'cache_B_misses_outer_product_m',
            'cache_B_total_accesses_outer_product_m',
            'cache_B_hit_rate_outer_product_m',
            'cache_B_hits_outer_product_n',
            'cache_B_misses_outer_product_n',
            'cache_B_total_accesses_outer_product_n',
            'cache_B_hit_rate_outer_product_n',
            'cache_B_hits_gustavsons_m',
            'cache_B_misses_gustavsons_m',
            'cache_B_total_accesses_gustavsons_m',
            'cache_B_hit_rate_gustavsons_m',
            'cache_B_hits_gustavsons_n',
            'cache_B_misses_gustavsons_n',
            'cache_B_total_accesses_gustavsons_n',
            'cache_B_hit_rate_gustavsons_n',

            # memory statistics
            'N_cycles_multiplying_inner_product_m',
            'N_cycles_merging_inner_product_m',
            'N_SRAM_weight_reads_inner_product_m',
            'N_SRAM_input_reads_inner_product_m',
            'N_SRAM_psum_reads_inner_product_m',
            'N_SRAM_psum_writes_inner_product_m',
            'N_DRAM_psum_writes_inner_product_m',
            'N_multiplications_inner_product_m',
            'N_cycles_multiplying_inner_product_n',
            'N_cycles_merging_inner_product_n',
            'N_SRAM_weight_reads_inner_product_n',
            'N_SRAM_input_reads_inner_product_n',
            'N_SRAM_psum_reads_inner_product_n',
            'N_SRAM_psum_writes_inner_product_n',
            'N_DRAM_psum_writes_inner_product_n',
            'N_multiplications_inner_product_n',
            'N_cycles_multiplying_outer_product_m',
            'N_cycles_merging_outer_product_m',
            'N_SRAM_weight_reads_outer_product_m',
            'N_SRAM_input_reads_outer_product_m',
            'N_SRAM_psum_reads_outer_product_m',
            'N_SRAM_psum_writes_outer_product_m',
            'N_DRAM_psum_writes_outer_product_m',
            'N_multiplications_outer_product_m',
            'N_cycles_multiplying_outer_product_n',
            'N_cycles_merging_outer_product_n',
            'N_SRAM_weight_reads_outer_product_n',
            'N_SRAM_input_reads_outer_product_n',
            'N_SRAM_psum_reads_outer_product_n',
            'N_SRAM_psum_writes_outer_product_n',
            'N_DRAM_psum_writes_outer_product_n',
            'N_multiplications_outer_product_n',
            'N_cycles_multiplying_gustavsons_m',
            'N_cycles_merging_gustavsons_m',
            'N_SRAM_weight_reads_gustavsons_m',
            'N_SRAM_input_reads_gustavsons_m',
            'N_SRAM_psum_reads_gustavsons_m',
            'N_SRAM_psum_writes_gustavsons_m',
            'N_DRAM_psum_writes_gustavsons_m',
            'N_multiplications_gustavsons_m',
            'N_cycles_multiplying_gustavsons_n',
            'N_cycles_merging_gustavsons_n',
            'N_SRAM_weight_reads_gustavsons_n',
            'N_SRAM_input_reads_gustavsons_n',
            'N_SRAM_psum_reads_gustavsons_n',
            'N_SRAM_psum_writes_gustavsons_n',
            'N_DRAM_psum_writes_gustavsons_n',
            'N_multiplications_gustavsons_n',
        ]), file=f)

        # insert all the data
        for i in range(len(layers)):
            print(';'.join([
                # general information
                layers[i].layer_name,
                str(layers[i].M),
                str(layers[i].N),
                str(layers[i].K),
                str(layers[i].A_nnz),
                str(layers[i].B_nnz),
                str(layers[i].A_sparsity),
                str(layers[i].B_sparsity),

                # cycles of each dataflow
                str(layers[i].inner_product_m.cycles),
                str(layers[i].inner_product_n.cycles),
                str(layers[i].outer_product_m.cycles),
                str(layers[i].outer_product_n.cycles),
                str(layers[i].gustavsons_m.cycles),
                str(layers[i].gustavsons_n.cycles),
                str(layers[i].best_cycles),
                str(layers[i].best_dataflow),

                # EC
                str(args.ec_cycles),

                # best dataflows mapping path considering EC
                str(mapping_with_EC[i].dataflow_name),
                str(mapping_with_EC[i].needs_EC),
                str(mapping_with_EC[i].EC_cycles if mapping_with_EC[i].EC_cycles > 0 else ''),
                str(mapping_with_EC[i].layer_cycles),
                str(mapping_with_EC[i].layer_cycles + mapping_with_EC[i].EC_cycles),
                str(mapping_with_EC[i].accum_cycles),

                # best dataflows mapping path avoiding EC
                str(mapping_without_EC[i].dataflow_name),
                str(mapping_without_EC[i].layer_cycles),
                str(mapping_without_EC[i].accum_cycles),

                # cache statistics
                str(layers[i].inner_product_m.cache_B_hits),
                str(layers[i].inner_product_m.cache_B_misses),
                str(layers[i].inner_product_m.cache_B_total_accesses),
                str(layers[i].inner_product_m.cache_B_hit_rate),
                str(layers[i].inner_product_n.cache_B_hits),
                str(layers[i].inner_product_n.cache_B_misses),
                str(layers[i].inner_product_n.cache_B_total_accesses),
                str(layers[i].inner_product_n.cache_B_hit_rate),
                str(layers[i].outer_product_m.cache_B_hits),
                str(layers[i].outer_product_m.cache_B_misses),
                str(layers[i].outer_product_m.cache_B_total_accesses),
                str(layers[i].outer_product_m.cache_B_hit_rate),
                str(layers[i].outer_product_n.cache_B_hits),
                str(layers[i].outer_product_n.cache_B_misses),
                str(layers[i].outer_product_n.cache_B_total_accesses),
                str(layers[i].outer_product_n.cache_B_hit_rate),
                str(layers[i].gustavsons_m.cache_B_hits),
                str(layers[i].gustavsons_m.cache_B_misses),
                str(layers[i].gustavsons_m.cache_B_total_accesses),
                str(layers[i].gustavsons_m.cache_B_hit_rate),
                str(layers[i].gustavsons_n.cache_B_hits),
                str(layers[i].gustavsons_n.cache_B_misses),
                str(layers[i].gustavsons_n.cache_B_total_accesses),
                str(layers[i].gustavsons_n.cache_B_hit_rate),

                # memory statistics
                str(layers[i].inner_product_m.N_cycles_multiplying),
                str(layers[i].inner_product_m.N_cycles_merging),
                str(layers[i].inner_product_m.N_SRAM_weight_reads),
                str(layers[i].inner_product_m.N_SRAM_input_reads),
                str(layers[i].inner_product_m.N_SRAM_psum_reads),
                str(layers[i].inner_product_m.N_SRAM_psum_writes),
                str(layers[i].inner_product_m.N_DRAM_psum_writes),
                str(layers[i].inner_product_m.N_multiplications),
                str(layers[i].inner_product_n.N_cycles_multiplying),
                str(layers[i].inner_product_n.N_cycles_merging),
                str(layers[i].inner_product_n.N_SRAM_weight_reads),
                str(layers[i].inner_product_n.N_SRAM_input_reads),
                str(layers[i].inner_product_n.N_SRAM_psum_reads),
                str(layers[i].inner_product_n.N_SRAM_psum_writes),
                str(layers[i].inner_product_n.N_DRAM_psum_writes),
                str(layers[i].inner_product_n.N_multiplications),
                str(layers[i].outer_product_m.N_cycles_multiplying),
                str(layers[i].outer_product_m.N_cycles_merging),
                str(layers[i].outer_product_m.N_SRAM_weight_reads),
                str(layers[i].outer_product_m.N_SRAM_input_reads),
                str(layers[i].outer_product_m.N_SRAM_psum_reads),
                str(layers[i].outer_product_m.N_SRAM_psum_writes),
                str(layers[i].outer_product_m.N_DRAM_psum_writes),
                str(layers[i].outer_product_m.N_multiplications),
                str(layers[i].outer_product_n.N_cycles_multiplying),
                str(layers[i].outer_product_n.N_cycles_merging),
                str(layers[i].outer_product_n.N_SRAM_weight_reads),
                str(layers[i].outer_product_n.N_SRAM_input_reads),
                str(layers[i].outer_product_n.N_SRAM_psum_reads),
                str(layers[i].outer_product_n.N_SRAM_psum_writes),
                str(layers[i].outer_product_n.N_DRAM_psum_writes),
                str(layers[i].outer_product_n.N_multiplications),
                str(layers[i].gustavsons_m.N_cycles_multiplying),
                str(layers[i].gustavsons_m.N_cycles_merging),
                str(layers[i].gustavsons_m.N_SRAM_weight_reads),
                str(layers[i].gustavsons_m.N_SRAM_input_reads),
                str(layers[i].gustavsons_m.N_SRAM_psum_reads),
                str(layers[i].gustavsons_m.N_SRAM_psum_writes),
                str(layers[i].gustavsons_m.N_DRAM_psum_writes),
                str(layers[i].gustavsons_m.N_multiplications),
                str(layers[i].gustavsons_n.N_cycles_multiplying),
                str(layers[i].gustavsons_n.N_cycles_merging),
                str(layers[i].gustavsons_n.N_SRAM_weight_reads),
                str(layers[i].gustavsons_n.N_SRAM_input_reads),
                str(layers[i].gustavsons_n.N_SRAM_psum_reads),
                str(layers[i].gustavsons_n.N_SRAM_psum_writes),
                str(layers[i].gustavsons_n.N_DRAM_psum_writes),
                str(layers[i].gustavsons_n.N_multiplications),
            ]), file=f)

    print(f"Model results saved to: {OUTPUT_FILE_FULLPATH}")
    

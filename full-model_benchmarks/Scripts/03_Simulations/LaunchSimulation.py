import os, sys, glob, shutil
import argparse
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SstStonneBenchmarkUtils import LayerDataFilenames

"""
Launches the different dataflows simulations for a given layer
It allows to specify an output directory to store all simulation results.
Also dataflows target can be specified to only launch some of them.

NOTE: this script requires BuildExecutionEnv.py to be executed before
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch simulations and parse results')
    parser.add_argument('layer_dir', type=str, help='Directory containing the information of the layer')
    parser.add_argument('-o', '--output', type=str, help='Output directory (default: same as input directory)')
    parser.add_argument('-f', '--force', action='store_true', help='Force to start again the simulation even if the results already exists in the directory')
    parser.add_argument('-t', '--target', choices=['IP-m', 'IP-n', 'OP-m', 'OP-n', 'Gust-m', 'Gust-n', 'all'], default='all', help='Simulation to execute')
    args = parser.parse_args()

    # get dirs
    WORKDIR = os.getcwd()
    OUTPUT_DIR = args.output if args.output is not None else args.layer_dir

    # get filenames
    FILENAMES = LayerDataFilenames.get_layer_data_filenames()
    TARGET_TO_DIR = {
        'IP-m': 'inner_product_m',
        'IP-n': 'inner_product_n',
        'OP-m': 'outer_product_m',
        'OP-n': 'outer_product_n',
        'Gust-m': 'gustavsons_m',
        'Gust-n': 'gustavsons_n',
    }

    # get targets
    targets = [args.target] if args.target != 'all' else ['IP-m', 'IP-n', 'OP-m', 'OP-n', 'Gust-m', 'Gust-n']

    print('Starting simulatations...')
    print('Targets selected:', targets)

    # execute all simulations
    for target in targets:
        # check if target folder exists
        if not os.path.exists(os.path.join(args.layer_dir, TARGET_TO_DIR[target])):
            print(f'[ERROR] Target {target} does not exist in folder {args.layer_dir}. Skipping...')
            continue

        print(f'[SIMULATION] Executing {target}...', end=' ')

        # get input and output directories (creating if not exists) for the target
        INPUT_DIR = os.path.abspath(os.path.join(args.layer_dir, TARGET_TO_DIR[target]))
        TARGET_OUTPUT_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, TARGET_TO_DIR[target]))
        os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)

        # check if there is results stored in the output dir
        if not args.force:
            if len(glob.glob(os.path.join(TARGET_OUTPUT_DIR, 'output*'))) > 0:
                print(f'Target {target} already executed (you can force it with -f). Skipping...')
                continue

        # execute the simulation, checking if it produces any output error
        os.chdir(INPUT_DIR)
        SST_SIM_FILE = FILENAMES[target + '_sst']
        result = subprocess.run(f"sst {SST_SIM_FILE}", shell=True, check=True)

        # if the simulation has been executed correctly...
        if result.returncode == 0:
            # clean old results that may produce some name aliasing
            for filename in glob.glob(os.path.join(TARGET_OUTPUT_DIR, 'output*')):
                try:
                    os.remove(filename)
                except FileNotFoundError:
                    pass

            # move the output files to the output directory
            for filename in glob.glob('output*') + [FILENAMES['mem_result']]:
                shutil.move(filename, os.path.join(TARGET_OUTPUT_DIR, filename))
            # copy the sst-sim file and the layer_info to the output directory
            try:
                shutil.copy(SST_SIM_FILE, os.path.join(TARGET_OUTPUT_DIR, SST_SIM_FILE))
                shutil.copy(FILENAMES['layer_info'], os.path.join(TARGET_OUTPUT_DIR, FILENAMES['layer_info']))
            except shutil.SameFileError:
                pass
        else:
            print(f'[ERROR] Target {target} failed. Skipping...')
            continue

        os.chdir(WORKDIR)
        print('[OK]')


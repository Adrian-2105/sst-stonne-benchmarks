import os, sys
import argparse
import json
import numpy as np
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SstStonneBenchmarkUtils import MemFileFormatLib as sstlib
from SstStonneBenchmarkUtils import LayerDataFilenames

"""
Given the root folder of a layer which contains the "layer_data" folder
(e.g., generated with ExtractDataLayer.py), checks if the results obtained
(for each dataflow and even for the original data contained in the layer folder)
after simulating the layer are correct.

The user can specify a DELTA (margin error) value. Also, if the results are
stored in a different folder than the original layer data folder, the user
can specify the path to the folder containing the results.
"""

DELTA = 0.0001

def results_equal_np(M, N, delta=DELTA):
    """
    Compares two numpy matrices considering a delta
    """
    if M.shape != N.shape:
        print(f'M.shape = {M.shape} != {N.shape} = N.shape', end=' ')
        return False

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isclose(M[i,j], N[i,j], atol=delta):
                print(f'M[{i}][{j}] = {M[i,j]} != {N[i,j]} = N[{i}][{j}]', end=' ')
                return False
    
    return True

def results_equal(M, N, delta=DELTA):
    if len(M) != len(N):
        print(f'len(M) = {len(M)} != {len(N)} = len(N)', end=' ')
        return False

    for i in range(len(M)):
        if not np.isclose(M[i], N[i], atol=delta):
            print(f'M[{i}] = {M[i]} != {N[i]} = N[{i}]', end=' ')
            return False
    
    return True


def calculate_correct_results(layer_dir):
    print(f"[INFO] Calculating correct results from raw-data for layer {layer_dir}...")

    # get input files
    FILES = LayerDataFilenames.get_layer_data_filenames(os.path.join(layer_dir, "layer_data"))
    # read layer information
    LAYER_DATA = json.load(open(FILES['layer_info'], 'r'))

    # read memory files 
    A_row_major_mem = sstlib.csv_read_file(FILES['A-row_mem'], unpack_data=True)
    B_row_major_mem = sstlib.csv_read_file(FILES['B-row_mem'], unpack_data=True)
    A_col_major_mem = sstlib.csv_read_file(FILES['A-col_mem'], unpack_data=True)
    B_col_major_mem = sstlib.csv_read_file(FILES['B-col_mem'], unpack_data=True)

    # first, check if all the mem files are present and generates the correct result
    # row bitmaps
    A_row_major_bitmap = sstlib.csv_read_file(FILES['A-row-major-bitmap'], unpack_data=False, use_int=True)
    A_matrix = sstlib.matrix_sparsebitmap2dense(A_row_major_bitmap, A_row_major_mem, LAYER_DATA['M'], LAYER_DATA['K'])
    B_row_major_bitmap = sstlib.csv_read_file(FILES['B-row-major-bitmap'], unpack_data=False, use_int=True)
    B_matrix = sstlib.matrix_sparsebitmap2dense(B_row_major_bitmap, B_row_major_mem, LAYER_DATA['K'], LAYER_DATA['N'])
    CORRECT_RESULT = A_matrix @ B_matrix
    print('[INFO] A (bitmap row-major) * B (bitmap row-major) [OK]')

    # col bitmaps
    A_col_major_bitmap = sstlib.csv_read_file(FILES['A-col-major-bitmap'], unpack_data=False, use_int=True)
    A_T_matrix = sstlib.matrix_sparsebitmap2dense(A_col_major_bitmap, A_col_major_mem, LAYER_DATA['K'], LAYER_DATA['M'])
    B_col_major_bitmap = sstlib.csv_read_file(FILES['B-col-major-bitmap'], unpack_data=False, use_int=True)
    B_T_matrix = sstlib.matrix_sparsebitmap2dense(B_col_major_bitmap, B_col_major_mem, LAYER_DATA['N'], LAYER_DATA['K'])
    C_T_matrix = B_T_matrix @ A_T_matrix
    assert(results_equal_np(CORRECT_RESULT.T, C_T_matrix, args.delta))
    print('[INFO] A (bitmap col-major) * B (bitmap col-major) [OK]')

    # csr
    A_csr_rowpointer = sstlib.csv_read_file(FILES['A-csr-rowp'], unpack_data=False, use_int=True)
    A_csr_colpointer = sstlib.csv_read_file(FILES['A-csr-colp'], unpack_data=False, use_int=True)
    A_matrix = sstlib.matrix_sparse2dense(A_csr_rowpointer, A_csr_colpointer, A_row_major_mem, LAYER_DATA['M'], LAYER_DATA['K'])
    B_csr_rowpointer = sstlib.csv_read_file(FILES['B-csr-rowp'], unpack_data=False, use_int=True)
    B_csr_colpointer = sstlib.csv_read_file(FILES['B-csr-colp'], unpack_data=False, use_int=True)
    B_matrix = sstlib.matrix_sparse2dense(B_csr_rowpointer, B_csr_colpointer, B_row_major_mem, LAYER_DATA['K'], LAYER_DATA['N'])
    C_matrix = A_matrix @ B_matrix
    assert(results_equal_np(CORRECT_RESULT, C_matrix, args.delta))
    print('[INFO] A (CSR) * B (CSR) [OK]')

    # csc
    A_csc_rowpointer = sstlib.csv_read_file(FILES['A-csc-rowp'], unpack_data=False, use_int=True)
    A_csc_colpointer = sstlib.csv_read_file(FILES['A-csc-colp'], unpack_data=False, use_int=True)
    A_T_matrix = sstlib.matrix_sparse2dense(A_csc_rowpointer, A_csc_colpointer, A_col_major_mem, LAYER_DATA['K'], LAYER_DATA['M'])
    B_csc_rowpointer = sstlib.csv_read_file(FILES['B-csc-rowp'], unpack_data=False, use_int=True)
    B_csc_colpointer = sstlib.csv_read_file(FILES['B-csc-colp'], unpack_data=False, use_int=True)
    B_T_matrix = sstlib.matrix_sparse2dense(B_csc_rowpointer, B_csc_colpointer, B_col_major_mem, LAYER_DATA['N'], LAYER_DATA['K'])
    C_T_matrix = B_T_matrix @ A_T_matrix
    assert(results_equal_np(CORRECT_RESULT.T, C_T_matrix, args.delta))
    print('[INFO] B (CSC) * A (CSC) [OK]')

    return CORRECT_RESULT


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check the results of the layer simulations')
    parser.add_argument('layer_dir', type=str, help='Directory containing the information of the layer')
    parser.add_argument('-r', '--results_dir', type=str, default="", help='Directory containing the results of the layer (same as layer_dir if not specified)')
    parser.add_argument('-d', '--delta', type=float, default=DELTA, help='Delta to consider')
    args = parser.parse_args()

    if args.results_dir == "": # same as layer dir
        args.results_dir = args.layer_dir

    # check if dir exists
    assert os.path.isdir(args.layer_dir), f"Directory {args.layer_dir} does not exist"

    # obtain the correct numpy results and transform them into mem arrays
    CORRECT_RESULT = calculate_correct_results(args.layer_dir)
    r, c, CORRECT_RESULT_MEM = sstlib.matrix_dense2sparse(CORRECT_RESULT)
    r, c, CORRECT_RESULT_T_MEM = sstlib.matrix_dense2sparse(CORRECT_RESULT.T)


    # iterate over all the results files
    for op in ['inner_product_m', 'outer_product_m', 'gustavsons_m', 'inner_product_n', 'outer_product_n', 'gustavsons_n']:
        # check if dir exists
        OP_DIR = os.path.join(args.results_dir, op)

        # check if the result have been obtained
        if len(glob.glob(os.path.join(OP_DIR, 'output*'))) == 0:
            print(f"[ERROR] Directory {OP_DIR} does not exists or does not contains any output file, skipping {op}...")
            continue

        # get output files
        FILES = LayerDataFilenames.get_layer_data_filenames(os.path.join(args.results_dir, op))


        # load layer info
        layer_info = json.load(open(FILES['layer_info'], 'r'))
        
        # read the result and transform it into a numpy matrix
        RESULT_MEM = sstlib.csv_read_file(FILES['mem_result'], unpack_data=False, use_int=False)
        if op.endswith('_m'):
            RESULT = sstlib.matrix_load_from_mem(RESULT_MEM, layer_info['M'], layer_info['N'])
        else:
            RESULT = sstlib.matrix_load_from_mem(RESULT_MEM, layer_info['N'], layer_info['M'])

        # compare the result with the correct one
        print(f"[INFO] Checking result of {op}... ", end='')
        # if op.endswith('_m') then is row-major format result, col-major otherwise
        if not results_equal_np(CORRECT_RESULT if op.endswith('_m') else CORRECT_RESULT.T, RESULT, args.delta):
            print(f"[FAILED]")
            continue

        print(f"[OK]")

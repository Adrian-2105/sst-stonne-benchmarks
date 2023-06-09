import os, sys
import argparse
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SstStonneBenchmarkUtils import MemFileFormatLib as sstlib
from SstStonneBenchmarkUtils import LayerDataFilenames

"""
This script generates a complete random layer given the dimensions of the matrices
A and B and their sparsity degrees. The layer is generated on the folder specified
by the user. The user can also specify if the layer should be generated even if
the folder already exists.

The generated folder will be completely prepared to be used by the other scripts
to generate its complete execution environment (02_EnvironmentGeneration/BuildExecutionEnv.py)
and to simulate the layer (03_Simulation/SimulateLayer.py).

You can also generate a complete model made out of random layers. Just ensure that all layers
have the same prefix and are numbered sequentially. In this case, layer name MUST end with the
layer number (e.g., layer_0, layer_1, layer_2, etc.)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a layer with random data.')
    parser.add_argument('--layer_dir', type=str, required=True, help='Path to the layer folder')
    parser.add_argument('--M', type=int, required=True, help='Number of rows of A')
    parser.add_argument('--K', type=int, required=True, help='Number of columns of A and rows of B')
    parser.add_argument('--N', type=int, required=True, help='Number of columns of B')
    parser.add_argument('--sparsityA', type=float, required=True, help='Sparsity (float[0, 1]) of the matrix A')
    parser.add_argument('--sparsityB', type=float, required=True, help='Sparsity (float[0, 1]) of the matrix B')
    parser.add_argument('--force', action='store_true', help='Force the generation of the layer')
    args = parser.parse_args()

    # check if the layer folder exists
    if os.path.isdir(args.layer_dir) and not args.force:
        print(f'Layer {args.layer_dir} already exists. Aborting layer creation...')
        exit(0)
    
    # create the layer folder
    TARGET_DIR = os.path.join(os.path.abspath(args.layer_dir), 'layer_data')
    print('Creating layer on folder', TARGET_DIR, '...')
    if not os.path.isdir(TARGET_DIR):
        os.makedirs(TARGET_DIR)



    ### Generating the input matrices of the layers

    # generate a random A matrix with numpy
    A = np.random.rand(args.M, args.K)
    # modify A to include the sparsity
    A[A >= args.sparsityA] = 0
    # obtain the CSR, CSC and bitmap (row-major and col-major) sparse version of A
    A_CSR_rowpointer, A_CSR_colpointer, A_CSR_data = sstlib.matrix_dense2sparse(A)
    A_CSC_rowpointer, A_CSC_colpointer, A_CSC_data = sstlib.matrix_dense2sparse(A.T)
    A_rowmajor_bitmap, A_rowmajor_bitmap_data = sstlib.matrix_dense2sparsebitmap(A)
    A_colmajor_bitmap, A_colmajor_bitmap_data = sstlib.matrix_dense2sparsebitmap(A.T)

    # generate a random B matrix with numpy
    B = np.random.rand(args.K, args.N)
    # modify B to include the sparsity
    B[B > args.sparsityB] = 0
    # obtain the CSR, CSC and bitmap (row-major and col-major) sparse version of B
    B_CSR_rowpointer, B_CSR_colpointer, B_CSR_data = sstlib.matrix_dense2sparse(B)
    B_CSC_rowpointer, B_CSC_colpointer, B_CSC_data = sstlib.matrix_dense2sparse(B.T)
    B_rowmajor_bitmap, B_rowmajor_bitmap_data = sstlib.matrix_dense2sparsebitmap(B)
    B_colmajor_bitmap, B_colmajor_bitmap_data = sstlib.matrix_dense2sparsebitmap(B.T)




    ### Generating the files of each matrix in each format (CSR, CSC, bitmap)

    # generate the layer files
    NEW_FILES = LayerDataFilenames.get_layer_data_filenames(TARGET_DIR)
    sstlib.csv_write_file(NEW_FILES['A-row-major-bitmap'], A_rowmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['A-csr-rowp'], A_CSR_rowpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['A-csr-colp'], A_CSR_colpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['A-row_mem'], A_rowmajor_bitmap_data, pack_data=True, use_int=True)

    sstlib.csv_write_file(NEW_FILES['A-col-major-bitmap'], A_colmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['A-csc-rowp'], A_CSC_rowpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['A-csc-colp'], A_CSC_colpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['A-col_mem'], A_colmajor_bitmap_data, pack_data=True, use_int=True)

    sstlib.csv_write_file(NEW_FILES['B-row-major-bitmap'], B_rowmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['B-csr-rowp'], B_CSR_rowpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['B-csr-colp'], B_CSR_colpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['B-row_mem'], B_rowmajor_bitmap_data, pack_data=True, use_int=True)

    sstlib.csv_write_file(NEW_FILES['B-col-major-bitmap'], B_colmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['B-csc-rowp'], B_CSC_rowpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['B-csc-colp'], B_CSC_colpointer, pack_data=False, use_int=True)
    sstlib.csv_write_file(NEW_FILES['B-col_mem'], B_colmajor_bitmap_data, pack_data=True, use_int=True)



    ### Adding other necessary files to the layer

    # copy the architectures
    TEMPLATE_ARCH_FILES = LayerDataFilenames.get_template_filenames(os.path.join(os.path.dirname((os.path.abspath(__file__))), 'HardwareConfs'))
    os.system(f'cp {TEMPLATE_ARCH_FILES["IP_arch"]} {NEW_FILES["IP_arch"]}')
    os.system(f'cp {TEMPLATE_ARCH_FILES["OP_arch"]} {NEW_FILES["OP_arch"]}')
    os.system(f'cp {TEMPLATE_ARCH_FILES["Gust_arch"]} {NEW_FILES["Gust_arch"]}')

    # build the JSON with the properties of the layer
    LAYER_DATA = {
        'name' : os.path.basename(args.layer_dir),
        'M' : args.M,
        'K' : args.K,
        'N' : args.N,
        'sparsity_A' : args.sparsityA,
        'sparsity_B' : args.sparsityB,
        'nnz_A' : len(A_CSR_data),
        'nnz_B' : len(B_CSR_data),
        'A_bytes' : len(A_CSR_data) * 4, # TODO: check if this is correct
        'B_bytes' : len(B_CSR_data) * 4, # TODO: check if this is correct
    }
    with open(NEW_FILES['layer_info'], 'w') as f:
        json.dump(LAYER_DATA, f, indent=4)





    ### Configuring the SST simulation files of each dataflow

    # copy the SST templates
    TEMPLATE_SST_FILES = LayerDataFilenames.get_template_filenames(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Templates'))
    os.system(f'cp {TEMPLATE_SST_FILES["IP_sst"]} {NEW_FILES["IP-m_sst"]}')
    os.system(f'cp {TEMPLATE_SST_FILES["IP_sst"]} {NEW_FILES["IP-n_sst"]}')
    os.system(f'cp {TEMPLATE_SST_FILES["OP_sst"]} {NEW_FILES["OP-m_sst"]}')
    os.system(f'cp {TEMPLATE_SST_FILES["OP_sst"]} {NEW_FILES["OP-n_sst"]}')
    os.system(f'cp {TEMPLATE_SST_FILES["Gust_sst"]} {NEW_FILES["Gust-m_sst"]}')
    os.system(f'cp {TEMPLATE_SST_FILES["Gust_sst"]} {NEW_FILES["Gust-n_sst"]}')

    # modify the templates according to the layer properties
    def set_sst_val(label, val, file, format_val=True):
        if format_val:
            if type(val) == str:
                val = val.split('/')[-1]
                val = f'"{val}"'
        sed = f'sed -i \'s/' # command start
        sed += f'\(\"{label}\" : \).*\(,\)/' # old text
        sed += f'\\1{val}\\2/' # new text
        sed += f'\' {file}' # file
        os.system(sed)

    # inner-product-m
    set_sst_val("GEMM_M", LAYER_DATA["M"], NEW_FILES['IP-m_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["N"], NEW_FILES['IP-m_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], NEW_FILES['IP-m_sst'])
    set_sst_val("matrix_a_dram_address", 0, NEW_FILES['IP-m_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["A_bytes"], NEW_FILES['IP-m_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], NEW_FILES['IP-m_sst'])
    set_sst_val("mem_init", NEW_FILES['mem_A-row_B-col'], NEW_FILES['IP-m_sst'])
    set_sst_val("bitmap_matrix_a_init", NEW_FILES['A-row-major-bitmap'], NEW_FILES['IP-m_sst'])
    set_sst_val("bitmap_matrix_b_init", NEW_FILES['B-row-major-bitmap'], NEW_FILES['IP-m_sst'])
    set_sst_val("hardware_configuration", NEW_FILES['IP_arch'], NEW_FILES['IP-m_sst'])
    # inner-product-n
    set_sst_val("GEMM_M", LAYER_DATA["N"], NEW_FILES['IP-n_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["M"], NEW_FILES['IP-n_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], NEW_FILES['IP-n_sst'])
    set_sst_val("matrix_a_dram_address", 0, NEW_FILES['IP-n_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["B_bytes"], NEW_FILES['IP-n_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], NEW_FILES['IP-n_sst'])
    set_sst_val("mem_init", NEW_FILES['mem_B-col_A-row'], NEW_FILES['IP-n_sst'])
    set_sst_val("bitmap_matrix_a_init", NEW_FILES['B-col-major-bitmap'], NEW_FILES['IP-n_sst'])
    set_sst_val("bitmap_matrix_b_init", NEW_FILES['A-col-major-bitmap'], NEW_FILES['IP-n_sst'])
    set_sst_val("hardware_configuration", NEW_FILES['IP_arch'], NEW_FILES['IP-n_sst'])
    # outer-product-m
    set_sst_val("GEMM_M", LAYER_DATA["M"], NEW_FILES['OP-m_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["N"], NEW_FILES['OP-m_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], NEW_FILES['OP-m_sst'])
    set_sst_val("matrix_a_dram_address", 0, NEW_FILES['OP-m_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["A_bytes"], NEW_FILES['OP-m_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], NEW_FILES['OP-m_sst'])
    set_sst_val("mem_init", NEW_FILES['mem_A-col_B-row'], NEW_FILES['OP-m_sst'])
    set_sst_val("rowpointer_matrix_a_init", NEW_FILES['A-csc-rowp'], NEW_FILES['OP-m_sst'])
    set_sst_val("colpointer_matrix_a_init", NEW_FILES['A-csc-colp'], NEW_FILES['OP-m_sst'])
    set_sst_val("rowpointer_matrix_b_init", NEW_FILES['B-csr-rowp'], NEW_FILES['OP-m_sst'])
    set_sst_val("colpointer_matrix_b_init", NEW_FILES['B-csr-colp'], NEW_FILES['OP-m_sst'])
    set_sst_val("hardware_configuration", NEW_FILES['OP_arch'], NEW_FILES['OP-m_sst'])
    # outer-product-n
    set_sst_val("GEMM_M", LAYER_DATA["N"], NEW_FILES['OP-n_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["M"], NEW_FILES['OP-n_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], NEW_FILES['OP-n_sst'])
    set_sst_val("matrix_a_dram_address", 0, NEW_FILES['OP-n_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["B_bytes"], NEW_FILES['OP-n_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], NEW_FILES['OP-n_sst'])
    set_sst_val("mem_init", NEW_FILES['mem_B-row_A-col'], NEW_FILES['OP-n_sst'])
    set_sst_val("rowpointer_matrix_a_init", NEW_FILES['B-csr-rowp'], NEW_FILES['OP-n_sst'])
    set_sst_val("colpointer_matrix_a_init", NEW_FILES['B-csr-colp'], NEW_FILES['OP-n_sst'])
    set_sst_val("rowpointer_matrix_b_init", NEW_FILES['A-csc-rowp'], NEW_FILES['OP-n_sst'])
    set_sst_val("colpointer_matrix_b_init", NEW_FILES['A-csc-colp'], NEW_FILES['OP-n_sst'])
    set_sst_val("hardware_configuration", NEW_FILES['OP_arch'], NEW_FILES['OP-n_sst'])
    # gustavsons-m
    set_sst_val("GEMM_M", LAYER_DATA["M"], NEW_FILES['Gust-m_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["N"], NEW_FILES['Gust-m_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], NEW_FILES['Gust-m_sst'])
    set_sst_val("matrix_a_dram_address", 0, NEW_FILES['Gust-m_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["A_bytes"], NEW_FILES['Gust-m_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], NEW_FILES['Gust-m_sst'])
    set_sst_val("mem_init", NEW_FILES['mem_A-row_B-row'], NEW_FILES['Gust-m_sst'])
    set_sst_val("rowpointer_matrix_a_init", NEW_FILES['A-csr-rowp'], NEW_FILES['Gust-m_sst'])
    set_sst_val("colpointer_matrix_a_init", NEW_FILES['A-csr-colp'], NEW_FILES['Gust-m_sst'])
    set_sst_val("rowpointer_matrix_b_init", NEW_FILES['B-csr-rowp'], NEW_FILES['Gust-m_sst'])
    set_sst_val("colpointer_matrix_b_init", NEW_FILES['B-csr-colp'], NEW_FILES['Gust-m_sst'])
    set_sst_val("hardware_configuration", NEW_FILES['Gust_arch'], NEW_FILES['Gust-m_sst'])
    # gustavsons-n
    set_sst_val("GEMM_M", LAYER_DATA["N"], NEW_FILES['Gust-n_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["M"], NEW_FILES['Gust-n_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], NEW_FILES['Gust-n_sst'])
    set_sst_val("matrix_a_dram_address", 0, NEW_FILES['Gust-n_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["B_bytes"], NEW_FILES['Gust-n_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], NEW_FILES['Gust-n_sst'])
    set_sst_val("mem_init", NEW_FILES['mem_B-col_A-col'], NEW_FILES['Gust-n_sst'])
    set_sst_val("rowpointer_matrix_a_init", NEW_FILES['B-csc-rowp'], NEW_FILES['Gust-n_sst'])
    set_sst_val("colpointer_matrix_a_init", NEW_FILES['B-csc-colp'], NEW_FILES['Gust-n_sst'])
    set_sst_val("rowpointer_matrix_b_init", NEW_FILES['A-csc-rowp'], NEW_FILES['Gust-n_sst'])
    set_sst_val("colpointer_matrix_b_init", NEW_FILES['A-csc-colp'], NEW_FILES['Gust-n_sst'])
    set_sst_val("hardware_configuration", NEW_FILES['Gust_arch'], NEW_FILES['Gust-n_sst'])





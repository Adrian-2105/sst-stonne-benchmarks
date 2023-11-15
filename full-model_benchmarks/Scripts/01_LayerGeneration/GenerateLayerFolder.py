import os, sys
import argparse
import re
import json
import torch
import numpy
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SstStonneBenchmarkUtils import MemFileFormatLib as sstlib
from SstStonneBenchmarkUtils import LayerDataFilenames

def patch_empty_rows(matrix : numpy.ndarray, row : bool):
    """
    Due to a STONNE bug, when running a inner-product simulation and the matrix
    has a row where all the elements are zero, the simulation crashes.

    This function patches that rows/columns which may cause that error by replacing
    a zero value with a very small value.
    """

    counter = 0

    if row:
        for i in range(matrix.shape[0]): 
            if numpy.count_nonzero(matrix[i,:]) == 0:
                counter += 1
                matrix[i][0] = 1e-5
    else: # column
        for i in range(matrix.shape[1]):
            if numpy.count_nonzero(matrix[:,i]) == 0:
                counter += 1
                matrix[0][i] = 1e-5

    return matrix, counter

def generate_layer_files(dir, input, weight):
    # Ensure that layer numbers matches
    assert(re.sub('\D+', '', input) == re.sub('\D+', '', weight))
    layer_idx = re.sub('\D+', '', input)

    print(f"Generating files for layer {layer_idx}...")

    # Read the input and weight tensors
    input_tensor = torch.load(os.path.join(dir, input)).detach().numpy()
    weight_tensor = torch.load(os.path.join(dir, weight)).detach().numpy()

    # Ensure that both tensors are 2D and share the same K dimension
    assert(len(input_tensor.shape) == 2)
    assert(len(weight_tensor.shape) == 2)
    assert(input_tensor.shape[1] == weight_tensor.shape[0])
    print(f"  Input shape: {input_tensor.shape}, weight shape: {weight_tensor.shape}")

    # Patch empty rows/columns to avoid SST-STONNE errors
    # Note: SST-STONNE crashes when running a inner-product and the LHS matrix has a empty row.
    #       Thus, we have to patch A empty rows and B empty columns (A * B -> B^T * A^T)
    input_tensor, counter_fix_input = patch_empty_rows(input_tensor, row=True)
    print(f"  Patched {counter_fix_input} empty rows in input tensor")
    weight_tensor, counter_fix_weight = patch_empty_rows(weight_tensor, row=False)
    print(f"  Patched {counter_fix_weight} empty columns in weight tensor")

    # Convert the tensors to bitmap, CSR and CSC
    input_rowmajor_bitmap, _ = sstlib.matrix_dense2sparsebitmap(input_tensor)
    input_colmajor_bitmap, _ = sstlib.matrix_dense2sparsebitmap(input_tensor.T)
    input_csr_rowp, input_csr_colp, input_csr_data = sstlib.matrix_dense2sparse(input_tensor)
    input_csc_rowp, input_csc_colp, input_csc_data = sstlib.matrix_dense2sparse(input_tensor.T)
    weight_rowmajor_bitmap, _ = sstlib.matrix_dense2sparsebitmap(weight_tensor)
    weight_colmajor_bitmap, _ = sstlib.matrix_dense2sparsebitmap(weight_tensor.T)
    weight_csr_rowp, weight_csr_colp, weight_csr_data = sstlib.matrix_dense2sparse(weight_tensor)
    weight_csc_rowp, weight_csc_colp, weight_csc_data = sstlib.matrix_dense2sparse(weight_tensor.T)

    # Generating layer folder
    LAYER_NAME = f"bench_{os.path.basename(dir)}_{layer_idx}"
    LAYER_FOLDER = os.path.join(dir, LAYER_NAME, "layer_data")
    os.makedirs(LAYER_FOLDER, exist_ok=True)
    FILENAMES = LayerDataFilenames.get_layer_data_filenames(basedir=LAYER_FOLDER)

    # build the JSON with the properties of the layer
    LAYER_DATA = {
        'name' : LAYER_NAME,
        'M' : input_tensor.shape[0],
        'K' : input_tensor.shape[1],
        'N' : weight_tensor.shape[1],
        'sparsity_A' : input_rowmajor_bitmap.count(0) / len(input_rowmajor_bitmap),
        'sparsity_B' : weight_rowmajor_bitmap.count(0) / len(weight_rowmajor_bitmap),
        'nnz_A' : len(input_csr_data),
        'nnz_B' : len(weight_csr_data),
        'A_bytes' : len(input_csr_data) * 4,
        'B_bytes' : len(weight_csr_data) * 4,
    }
    with open(FILENAMES['layer_info'], 'w') as f:
        json.dump(LAYER_DATA, f, indent=4)

    # Generate the layer files
    sstlib.csv_write_file(FILENAMES["A-row-major-bitmap"], input_rowmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["A-csr-rowp"], input_csr_rowp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["A-csr-colp"], input_csr_colp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["A-row_mem"], input_csr_data, pack_data=True)
    sstlib.csv_write_file(FILENAMES["A-col-major-bitmap"], input_colmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["A-csc-rowp"], input_csc_rowp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["A-csc-colp"], input_csc_colp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["A-col_mem"], input_csc_data, pack_data=True)

    sstlib.csv_write_file(FILENAMES["B-row-major-bitmap"], weight_rowmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["B-csr-rowp"], weight_csr_rowp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["B-csr-colp"], weight_csr_colp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["B-row_mem"], weight_csr_data, pack_data=True)
    sstlib.csv_write_file(FILENAMES["B-col-major-bitmap"], weight_colmajor_bitmap, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["B-csc-rowp"], weight_csc_rowp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["B-csc-colp"], weight_csc_colp, pack_data=False, use_int=True)
    sstlib.csv_write_file(FILENAMES["B-col_mem"], weight_csc_data, pack_data=True)

    # Copy hardware configs templates
    HARDWARE_CONFS = LayerDataFilenames.get_template_filenames(basedir=os.path.join(os.path.dirname(__file__), "HardwareConfs"))
    shutil.copyfile(HARDWARE_CONFS["IP_arch"], FILENAMES["IP_arch"])
    shutil.copyfile(HARDWARE_CONFS["OP_arch"], FILENAMES["OP_arch"])
    shutil.copyfile(HARDWARE_CONFS["Gust_arch"], FILENAMES["Gust_arch"])

    # Copy SST templates
    TEMPLATES = LayerDataFilenames.get_template_filenames(basedir=os.path.join(os.path.dirname(__file__), "Templates"))
    shutil.copyfile(TEMPLATES["IP_sst"], FILENAMES["IP-m_sst"])
    shutil.copyfile(TEMPLATES["IP_sst"], FILENAMES["IP-n_sst"])
    shutil.copyfile(TEMPLATES["OP_sst"], FILENAMES["OP-m_sst"])
    shutil.copyfile(TEMPLATES["OP_sst"], FILENAMES["OP-n_sst"])
    shutil.copyfile(TEMPLATES["Gust_sst"], FILENAMES["Gust-m_sst"])
    shutil.copyfile(TEMPLATES["Gust_sst"], FILENAMES["Gust-n_sst"])

    # Configuring parameters of SST files
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
    set_sst_val("hardware_configuration", FILENAMES['IP_arch'], FILENAMES['IP-m_sst'])
    set_sst_val("GEMM_M", LAYER_DATA["M"], FILENAMES['IP-m_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["N"], FILENAMES['IP-m_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], FILENAMES['IP-m_sst'])
    set_sst_val("matrix_a_dram_address", 0, FILENAMES['IP-m_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["A_bytes"], FILENAMES['IP-m_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], FILENAMES['IP-m_sst'])
    set_sst_val("mem_init", FILENAMES['mem_A-row_B-col'], FILENAMES['IP-m_sst'])
    set_sst_val("bitmap_matrix_a_init", FILENAMES['A-row-major-bitmap'], FILENAMES['IP-m_sst'])
    set_sst_val("bitmap_matrix_b_init", FILENAMES['B-row-major-bitmap'], FILENAMES['IP-m_sst'])
    # inner-product-n
    set_sst_val("hardware_configuration", FILENAMES['IP_arch'], FILENAMES['IP-n_sst'])
    set_sst_val("GEMM_M", LAYER_DATA["N"], FILENAMES['IP-n_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["M"], FILENAMES['IP-n_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], FILENAMES['IP-n_sst'])
    set_sst_val("matrix_a_dram_address", 0, FILENAMES['IP-n_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["B_bytes"], FILENAMES['IP-n_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], FILENAMES['IP-n_sst'])
    set_sst_val("mem_init", FILENAMES['mem_B-col_A-row'], FILENAMES['IP-n_sst'])
    set_sst_val("bitmap_matrix_a_init", FILENAMES['B-col-major-bitmap'], FILENAMES['IP-n_sst'])
    set_sst_val("bitmap_matrix_b_init", FILENAMES['A-col-major-bitmap'], FILENAMES['IP-n_sst'])
    # outer-product-m
    set_sst_val("hardware_configuration", FILENAMES['OP_arch'], FILENAMES['OP-m_sst'])
    set_sst_val("GEMM_M", LAYER_DATA["M"], FILENAMES['OP-m_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["N"], FILENAMES['OP-m_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], FILENAMES['OP-m_sst'])
    set_sst_val("matrix_a_dram_address", 0, FILENAMES['OP-m_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["A_bytes"], FILENAMES['OP-m_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], FILENAMES['OP-m_sst'])
    set_sst_val("mem_init", FILENAMES['mem_A-col_B-row'], FILENAMES['OP-m_sst'])
    set_sst_val("rowpointer_matrix_a_init", FILENAMES['A-csc-rowp'], FILENAMES['OP-m_sst'])
    set_sst_val("colpointer_matrix_a_init", FILENAMES['A-csc-colp'], FILENAMES['OP-m_sst'])
    set_sst_val("rowpointer_matrix_b_init", FILENAMES['B-csr-rowp'], FILENAMES['OP-m_sst'])
    set_sst_val("colpointer_matrix_b_init", FILENAMES['B-csr-colp'], FILENAMES['OP-m_sst'])
    # outer-product-n
    set_sst_val("hardware_configuration", FILENAMES['OP_arch'], FILENAMES['OP-n_sst'])
    set_sst_val("GEMM_M", LAYER_DATA["N"], FILENAMES['OP-n_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["M"], FILENAMES['OP-n_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], FILENAMES['OP-n_sst'])
    set_sst_val("matrix_a_dram_address", 0, FILENAMES['OP-n_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["B_bytes"], FILENAMES['OP-n_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], FILENAMES['OP-n_sst'])
    set_sst_val("mem_init", FILENAMES['mem_B-row_A-col'], FILENAMES['OP-n_sst'])
    set_sst_val("rowpointer_matrix_a_init", FILENAMES['B-csr-rowp'], FILENAMES['OP-n_sst'])
    set_sst_val("colpointer_matrix_a_init", FILENAMES['B-csr-colp'], FILENAMES['OP-n_sst'])
    set_sst_val("rowpointer_matrix_b_init", FILENAMES['A-csc-rowp'], FILENAMES['OP-n_sst'])
    set_sst_val("colpointer_matrix_b_init", FILENAMES['A-csc-colp'], FILENAMES['OP-n_sst'])
    # gustavsons-m
    set_sst_val("hardware_configuration", FILENAMES['Gust_arch'], FILENAMES['Gust-m_sst'])
    set_sst_val("GEMM_M", LAYER_DATA["M"], FILENAMES['Gust-m_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["N"], FILENAMES['Gust-m_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], FILENAMES['Gust-m_sst'])
    set_sst_val("matrix_a_dram_address", 0, FILENAMES['Gust-m_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["A_bytes"], FILENAMES['Gust-m_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], FILENAMES['Gust-m_sst'])
    set_sst_val("mem_init", FILENAMES['mem_A-row_B-row'], FILENAMES['Gust-m_sst'])
    set_sst_val("rowpointer_matrix_a_init", FILENAMES['A-csr-rowp'], FILENAMES['Gust-m_sst'])
    set_sst_val("colpointer_matrix_a_init", FILENAMES['A-csr-colp'], FILENAMES['Gust-m_sst'])
    set_sst_val("rowpointer_matrix_b_init", FILENAMES['B-csr-rowp'], FILENAMES['Gust-m_sst'])
    set_sst_val("colpointer_matrix_b_init", FILENAMES['B-csr-colp'], FILENAMES['Gust-m_sst'])
    # gustavsons-n
    set_sst_val("hardware_configuration", FILENAMES['Gust_arch'], FILENAMES['Gust-n_sst'])
    set_sst_val("GEMM_M", LAYER_DATA["N"], FILENAMES['Gust-n_sst'])
    set_sst_val("GEMM_N", LAYER_DATA["M"], FILENAMES['Gust-n_sst'])
    set_sst_val("GEMM_K", LAYER_DATA["K"], FILENAMES['Gust-n_sst'])
    set_sst_val("matrix_a_dram_address", 0, FILENAMES['Gust-n_sst'])
    set_sst_val("matrix_b_dram_address", LAYER_DATA["B_bytes"], FILENAMES['Gust-n_sst'])
    set_sst_val("matrix_c_dram_address", LAYER_DATA["A_bytes"] + LAYER_DATA["B_bytes"], FILENAMES['Gust-n_sst'])
    set_sst_val("mem_init", FILENAMES['mem_B-col_A-col'], FILENAMES['Gust-n_sst'])
    set_sst_val("rowpointer_matrix_a_init", FILENAMES['B-csc-rowp'], FILENAMES['Gust-n_sst'])
    set_sst_val("colpointer_matrix_a_init", FILENAMES['B-csc-colp'], FILENAMES['Gust-n_sst'])
    set_sst_val("rowpointer_matrix_b_init", FILENAMES['A-csc-rowp'], FILENAMES['Gust-n_sst'])
    set_sst_val("colpointer_matrix_b_init", FILENAMES['A-csc-colp'], FILENAMES['Gust-n_sst'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates the initial files for a model given the PyTorch tensors folder')
    parser.add_argument('-d', '--dir', required=True, help='Directory where all the PyTorch tensors of the model are located')
    args = parser.parse_args()

    # Read all the files in the directory
    files = os.listdir(args.dir)
    inputs = [x for x in files if x.startswith("input")]
    weights = [x for x in files if x.startswith("weight")]
    assert(len(inputs) == len(weights))

    # Sort the filenames by the number in the filename (example: input_1_linear_lhs.pt)
    inputs.sort(key=lambda f: int(re.sub('\D+', '', f)))
    weights.sort(key=lambda f: int(re.sub('\D+', '', f)))

    for i, w in zip(inputs, weights):
        generate_layer_files(args.dir, i, w)

    
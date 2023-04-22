import os, sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SstStonneBenchmarkUtils import MemFileFormatLib as sstlib
from SstStonneBenchmarkUtils import LayerDataFilenames

"""
Given the root folder of a layer which contains the "layer_data" folder
(e.g., generated with ExtractDataLayer.py), prepares the complete execution
environment for the layer, including the SST Python scripts, the mem files,
the matrix files, etc.

The environment is prepared for each dataflow (inner-product-m, inner-product-n,
outer-product-m, outer-product-n, gustavsons-m, gustavsons-n) in separates folders
inside the layer folder, all of them ready to execute the SST simulation.

After generating it, the dataflow can be executed with the following command:
   {layerFolder/dataflowFolder} -> sst sst_{script-name}.py

NOTE: this implementation has been made to reduce the number of files in the
      layer folder (and in the repository itself). Do not upload the generated
      files to the repository. 
"""

def copy_file(src, dst):
    # check if src file exists
    if not os.path.isfile(src):
        print(f"[WARNING] File {src} does not exist, cannot be copied to {dst}")
    os.system(f"cp {src} {dst}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts all useful information from a layer')
    parser.add_argument('layer_dir', help='Directory containing the information of the layer')
    args = parser.parse_args()

    # check if dir exists
    assert os.path.isdir(args.layer_dir), f"Directory {args.layer_dir} does not exist"
    # check if dir contains the needed dirs
    assert os.path.isdir(os.path.join(args.layer_dir, "layer_data")), f"Directory {args.layer_dir} does not contain layer_data"

    # create dirs to build the environments
    def create_dir(dir_name):
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
    IP_M_FOLDER = os.path.join(args.layer_dir, "inner_product_m")
    IP_N_FOLDER = os.path.join(args.layer_dir, "inner_product_n")
    OP_M_FOLDER = os.path.join(args.layer_dir, "outer_product_m")
    OP_N_FOLDER = os.path.join(args.layer_dir, "outer_product_n")
    GUST_M_FOLDER = os.path.join(args.layer_dir, "gustavsons_m")
    GUST_N_FOLDER = os.path.join(args.layer_dir, "gustavsons_n")
    for folder in [IP_M_FOLDER, IP_N_FOLDER, OP_M_FOLDER, OP_N_FOLDER, GUST_M_FOLDER, GUST_N_FOLDER]:
        create_dir(folder)


    # get input files
    FILES = LayerDataFilenames.get_layer_data_filenames(os.path.join(args.layer_dir, "layer_data"))

    # copy files from the folders
    try:
        # inner-product-m
        for file in [FILES['A-row-major-bitmap'], FILES['B-row-major-bitmap'], FILES['IP_arch'], FILES['IP-m_sst']]:
            copy_file(file, IP_M_FOLDER)
        # inner-product-n
        for file in [FILES['A-col-major-bitmap'], FILES['B-col-major-bitmap'], FILES['IP_arch'], FILES['IP-n_sst']]:
            copy_file(file, IP_N_FOLDER)

        # outer-product-m
        for file in [FILES['A-csc-rowp'], FILES['A-csc-colp'], FILES['B-csr-rowp'], FILES['B-csr-colp'], FILES['OP_arch'], FILES['OP-m_sst']]:
            copy_file(file, OP_M_FOLDER)
        # outer-product-n
        for file in [FILES['A-csc-rowp'], FILES['A-csc-colp'], FILES['B-csr-rowp'], FILES['B-csr-colp'], FILES['OP_arch'], FILES['OP-n_sst']]:
            copy_file(file, OP_N_FOLDER)

        # gustavsons-m
        for file in [FILES['A-csr-rowp'], FILES['A-csr-colp'], FILES['B-csr-rowp'], FILES['B-csr-colp'], FILES['Gust_arch'], FILES['Gust-m_sst']]:
            copy_file(file, GUST_M_FOLDER)
        # gustavsons-n
        for file in [FILES['A-csc-rowp'], FILES['A-csc-colp'], FILES['B-csc-rowp'], FILES['B-csc-colp'], FILES['Gust_arch'], FILES['Gust-n_sst']]:
            copy_file(file, GUST_N_FOLDER)

        # generate mem initialization files for each dataflow
        def generate_mem_init_file(A_mem_file, B_mem_file, output_mem_file):
            try:
                A_mem = sstlib.csv_read_file(A_mem_file, unpack_data=False, use_int=True)
                B_mem = sstlib.csv_read_file(B_mem_file, unpack_data=False, use_int=True)
                merged_mem = sstlib.merge_mem(A_mem, B_mem)
                sstlib.csv_write_file(output_mem_file, merged_mem, pack_data=False, use_int=True)
            except Exception as e:
                print(f"[WARNING] Could not generate mem init {output_mem_file} file"
                      f"because {A_mem_file} and/or {B_mem_file} mem files are missing: {e}")
                
        
        # inner-product-m
        generate_mem_init_file(FILES['A-row_mem'], FILES['B-col_mem'], os.path.join(IP_M_FOLDER, os.path.basename(FILES['mem_A-row_B-col'])))
        # inner-product-n
        generate_mem_init_file(FILES['B-col_mem'], FILES['A-row_mem'], os.path.join(IP_N_FOLDER, os.path.basename(FILES['mem_B-col_A-row'])))
        # outer-product-m
        generate_mem_init_file(FILES['A-col_mem'], FILES['B-row_mem'], os.path.join(OP_M_FOLDER, os.path.basename(FILES['mem_A-col_B-row'])))
        # outer-product-n
        generate_mem_init_file(FILES['B-row_mem'], FILES['A-col_mem'], os.path.join(OP_N_FOLDER, os.path.basename(FILES['mem_B-row_A-col'])))
        # gustavsons-m
        generate_mem_init_file(FILES['A-row_mem'], FILES['B-row_mem'], os.path.join(GUST_M_FOLDER, os.path.basename(FILES['mem_A-row_B-row'])))
        # gustavsons-n
        generate_mem_init_file(FILES['B-col_mem'], FILES['A-col_mem'], os.path.join(GUST_N_FOLDER, os.path.basename(FILES['mem_B-col_A-col'])))


    except Exception as e:
        print(e.__traceback__.tb_lineno, e)
        print("FATAL ERROR: Error while extracting data from layer")
        exit(1)

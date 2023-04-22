import os

def get_layer_data_filenames(basedir=""):
    LAYER_DATA_FILES = {
        "IP-m_sst" :      os.path.join(basedir, 'sst_stonne_inner_product_m.py'),
        "IP-n_sst" :      os.path.join(basedir, 'sst_stonne_inner_product_n.py'),
        "IP_arch" :       os.path.join(basedir, 'sigma_64mses_64_bw.cfg'),

        "OP-m_sst" :      os.path.join(basedir, 'sst_stonne_outer_product_m.py'),
        "OP-n_sst" :      os.path.join(basedir, 'sst_stonne_outer_product_n.py'),
        "OP_arch" :       os.path.join(basedir, 'sparseflex_op_64mses_64_bw.cfg'),

        "Gust-m_sst" :    os.path.join(basedir, 'sst_stonne_gustavsons_m.py'),
        "Gust-n_sst" :    os.path.join(basedir, 'sst_stonne_gustavsons_n.py'),
        "Gust_arch" :     os.path.join(basedir, 'sparseflex_gustavsons_64mses_64_bw.cfg'),

        "A-row-major-bitmap" :  os.path.join(basedir, 'A_row-major_bitmap.in'),
        "A-csr-rowp" :    os.path.join(basedir, 'A_csr_rowpointer.in'),
        "A-csr-colp" :    os.path.join(basedir, 'A_csr_colpointer.in'),
        "A-row_mem" :     os.path.join(basedir, 'A_row-major_mem.ini'),

        "A-col-major-bitmap" :  os.path.join(basedir, 'A_col-major_bitmap.in'),
        "A-csc-rowp" :    os.path.join(basedir, 'A_csc_rowpointer.in'),
        "A-csc-colp" :    os.path.join(basedir, 'A_csc_colpointer.in'),
        "A-col_mem" :     os.path.join(basedir, 'A_col-major_mem.ini'),

        "B-row-major-bitmap" :  os.path.join(basedir, 'B_row-major_bitmap.in'),
        "B-csr-rowp" :    os.path.join(basedir, 'B_csr_rowpointer.in'),
        "B-csr-colp" :    os.path.join(basedir, 'B_csr_colpointer.in'),
        "B-row_mem" :     os.path.join(basedir, 'B_row-major_mem.ini'),

        "B-col-major-bitmap" :  os.path.join(basedir, 'B_col-major_bitmap.in'),
        "B-csc-rowp" :    os.path.join(basedir, 'B_csc_rowpointer.in'),
        "B-csc-colp" :    os.path.join(basedir, 'B_csc_colpointer.in'),
        "B-col_mem" :     os.path.join(basedir, 'B_col-major_mem.ini'),

        "mem_A-row_B-row": os.path.join(basedir, 'mem_A-row_B-row.ini'),
        "mem_A-row_B-col": os.path.join(basedir, 'mem_A-row_B-col.ini'),
        "mem_A-col_B-row": os.path.join(basedir, 'mem_A-col_B-row.ini'),
        #"mem_A-col_B-col": os.path.join(basedir, 'mem_A-col_B-col.ini'),
        #"mem_B-row_A-row": os.path.join(basedir, 'mem_B-row_A-row.ini'),
        "mem_B-row_A-col": os.path.join(basedir, 'mem_B-row_A-col.ini'),
        "mem_B-col_A-row": os.path.join(basedir, 'mem_B-col_A-row.ini'),
        "mem_B-col_A-col": os.path.join(basedir, 'mem_B-col_A-col.ini'),

        "layer_info" :    os.path.join(basedir, 'layer_info.json'),
        "mem_result" :    os.path.join(basedir, 'result.out'),
    }
    return LAYER_DATA_FILES

def get_old_filenames(basedir=""):
    OLD_FILES = {
        "IP-m_A-bitmap" : os.path.join(basedir, 'inner_product_m', 'bitmapSpMSpM_file_bitmapA.in'),
        "IP-m_B-bitmap" : os.path.join(basedir, 'inner_product_m', 'bitmapSpMSpM_file_bitmapB.in'),
        "IP-m_mem" :      os.path.join(basedir, 'inner_product_m', 'bitmapSpMSpM_gemm_mem.ini'),
        "IP-m_sst" :      os.path.join(basedir, 'inner_product_m', 'temporal_sst_stonne_bitmapSpMSpM.py'),
        "IP-m_arch" :     os.path.join(basedir, 'inner_product_m', 'sigma_128mses_128_bw.cfg'),

        "OP-m_A-rowp" :   os.path.join(basedir, 'outer_product_m', 'outerproduct_gemm_rowpointerA.in'),
        "OP-m_A-colp" :   os.path.join(basedir, 'outer_product_m', 'outerproduct_gemm_colpointerA.in'),
        "OP-m_B-rowp" :   os.path.join(basedir, 'outer_product_m', 'outerproduct_gemm_rowpointerB.in'),
        "OP-m_B-colp" :   os.path.join(basedir, 'outer_product_m', 'outerproduct_gemm_colpointerB.in'),
        "OP-m_mem" :      os.path.join(basedir, 'outer_product_m', 'outerproduct_gemm_mem.ini'),
        "OP-m_sst" :      os.path.join(basedir, 'outer_product_m', 'temporal_sst_stonne_outerProduct.py'),
        "OP-m_arch" :     os.path.join(basedir, 'outer_product_m', 'sparseflex_op_128mses_128_bw.cfg'),

        "Gust-m_A-rowp" : os.path.join(basedir, 'gustavsons_m', 'gustavsons_gemm_rowpointerA.in'),
        "Gust-m_A-colp" : os.path.join(basedir, 'gustavsons_m', 'gustavsons_gemm_colpointerA.in'),
        "Gust-m_mem" :    os.path.join(basedir, 'gustavsons_m', 'gustavsons_gemm_mem.ini'),
        "Gust-m_sst" :    os.path.join(basedir, 'gustavsons_m', 'temporal_sst_stonne_gustavsons.py'),
        "Gust-m_arch" :   os.path.join(basedir, 'gustavsons_m', 'sparseflex_gustavsons_128mses_128_bw.cfg'),
    }
    return OLD_FILES

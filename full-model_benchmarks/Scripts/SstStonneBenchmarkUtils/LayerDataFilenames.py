import os

def get_layer_data_filenames(basedir=""):
    """
    Returns a dict with all the names configured for the layer data files.
    All the scripts in this testbench MUST use this function to obtain the
    filenames in order to avoid hardcoding the filenames in the scripts
    (and thus maintain coherence between all the scripts)

    User can specify a base directory to be prepended to all the filenames
    """

    LAYER_DATA_FILES = {
        # SST Files
        "IP-m_sst" :            'sst_stonne_inner_product_m.py',
        "IP-n_sst" :            'sst_stonne_inner_product_n.py',
        "OP-m_sst" :            'sst_stonne_outer_product_m.py',
        "OP-n_sst" :            'sst_stonne_outer_product_n.py',
        "Gust-m_sst" :          'sst_stonne_gustavsons_m.py',
        "Gust-n_sst" :          'sst_stonne_gustavsons_n.py',

        # Arch Config Files
        "IP_arch" :             'sigma_64mses_64_bw.cfg',
        "OP_arch" :             'sparseflex_op_64mses_64_bw.cfg',
        "Gust_arch" :           'sparseflex_gustavsons_64mses_64_bw.cfg',

        # A Row Major files
        "A-row-major-bitmap" :  'A_row-major_bitmap.in',
        "A-csr-rowp" :          'A_csr_rowpointer.in',
        "A-csr-colp" :          'A_csr_colpointer.in',
        "A-row_mem" :           'A_row-major_mem.ini',
        # A Col Major files
        "A-col-major-bitmap" :  'A_col-major_bitmap.in',
        "A-csc-rowp" :          'A_csc_rowpointer.in',
        "A-csc-colp" :          'A_csc_colpointer.in',
        "A-col_mem" :           'A_col-major_mem.ini',

        # B Row Major files
        "B-row-major-bitmap" :  'B_row-major_bitmap.in',
        "B-csr-rowp" :          'B_csr_rowpointer.in',
        "B-csr-colp" :          'B_csr_colpointer.in',
        "B-row_mem" :           'B_row-major_mem.ini',
        # B Col Major files
        "B-col-major-bitmap" :  'B_col-major_bitmap.in',
        "B-csc-rowp" :          'B_csc_rowpointer.in',
        "B-csc-colp" :          'B_csc_colpointer.in',
        "B-col_mem" :           'B_col-major_mem.ini',

        # Mem Init (A + B) Files
        "mem_A-row_B-row":      'mem_A-row_B-row.ini',
        "mem_A-row_B-col":      'mem_A-row_B-col.ini',
        "mem_A-col_B-row":      'mem_A-col_B-row.ini',
        #"mem_A-col_B-col":      'mem_A-col_B-col.ini'),
        #"mem_B-row_A-row":      'mem_B-row_A-row.ini'),
        "mem_B-row_A-col":      'mem_B-row_A-col.ini',
        "mem_B-col_A-row":      'mem_B-col_A-row.ini',
        "mem_B-col_A-col":      'mem_B-col_A-col.ini',

        # Misc Files
        "layer_info" :          'layer_info.json',
        "mem_result" :          'result.out',
    }

    return {key: os.path.join(basedir, value) for key, value in LAYER_DATA_FILES.items()}


def get_template_filenames(basedir=""):
    """
    Returns a dict with all the names configured for the template files.

    User can specify a base directory to be prepended to all the filenames
    """

    TEMPLATE_FILENAMES = {
        # SST Files
        "IP_sst" :      'template_sst_stonne_inner_product.py',
        "OP_sst" :      'template_sst_stonne_outer_product.py',
        "Gust_sst" :    'template_sst_stonne_gustavsons.py',

        # Arch Config Files
        "IP_arch" :     'sigma_64mses_64_bw.cfg',
        "OP_arch" :     'sparseflex_op_64mses_64_bw.cfg',
        "Gust_arch" :   'sparseflex_gustavsons_64mses_64_bw.cfg'
    }

    return {key: os.path.join(basedir, value) for key, value in TEMPLATE_FILENAMES.items()}


def get_old_filenames(basedir=""):
    """
    This is an OLD VERSION of the filenames that were used in the original testbench
    used for the Flexagon paper. These filenames are not used anymore, only for 
    translating the old testbench to the new one.

    User can specify a base directory to be prepended to all the filenames
    """

    OLD_FILES = {
        # Inner-Product M Files
        "IP-m_A-bitmap" : os.path.join('inner_product_m', 'bitmapSpMSpM_file_bitmapA.in'),
        "IP-m_B-bitmap" : os.path.join('inner_product_m', 'bitmapSpMSpM_file_bitmapB.in'),
        "IP-m_mem" :      os.path.join('inner_product_m', 'bitmapSpMSpM_gemm_mem.ini'),
        "IP-m_sst" :      os.path.join('inner_product_m', 'temporal_sst_stonne_bitmapSpMSpM.py'),
        "IP-m_arch" :     os.path.join('inner_product_m', 'sigma_64mses_64_bw.cfg'),

        # Outer-Product M Files
        "OP-m_A-rowp" :   os.path.join('outer_product_m', 'outerproduct_gemm_rowpointerA.in'),
        "OP-m_A-colp" :   os.path.join('outer_product_m', 'outerproduct_gemm_colpointerA.in'),
        "OP-m_B-rowp" :   os.path.join('outer_product_m', 'outerproduct_gemm_rowpointerB.in'),
        "OP-m_B-colp" :   os.path.join('outer_product_m', 'outerproduct_gemm_colpointerB.in'),
        "OP-m_mem" :      os.path.join('outer_product_m', 'outerproduct_gemm_mem.ini'),
        "OP-m_sst" :      os.path.join('outer_product_m', 'temporal_sst_stonne_outerProduct.py'),
        "OP-m_arch" :     os.path.join('outer_product_m', 'sparseflex_op_64mses_64_bw.cfg'),

        # Gustavsons M Files
        "Gust-m_A-rowp" : os.path.join('gustavsons_m', 'gustavsons_gemm_rowpointerA.in'),
        "Gust-m_A-colp" : os.path.join('gustavsons_m', 'gustavsons_gemm_colpointerA.in'),
        "Gust-m_B-rowp" : os.path.join('gustavsons_m', 'gustavsons_gemm_rowpointerB.in'),
        "Gust-m_B-colp" : os.path.join('gustavsons_m', 'gustavsons_gemm_colpointerB.in'),
        "Gust-m_mem" :    os.path.join('gustavsons_m', 'gustavsons_gemm_mem.ini'),
        "Gust-m_sst" :    os.path.join('gustavsons_m', 'temporal_sst_stonne_gustavsons.py'),
        "Gust-m_arch" :   os.path.join('gustavsons_m', 'sparseflex_gustavsons_64mses_64_bw.cfg'),
    }

    return {key: os.path.join(basedir, value) for key, value in OLD_FILES.items()}

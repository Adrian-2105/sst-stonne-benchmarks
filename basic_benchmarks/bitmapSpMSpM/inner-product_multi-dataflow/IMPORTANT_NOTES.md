TAKE INTO ACCOUNT ABOUT THE REPRESENTATION:


INNER PRODUCT M

    A Matrix (1st matrix) requires:
    - bitmap matrix with row-major format -> `A-row_bitmap.in`
    - mem values array with values stored in row-major format

    B Matrix (2nd matrix) requires:
    - bitmap matrix with row-major format -> `B-row_bitmap.in`
    - mem values array with values stored in **col-major** format

    Mem file -> `A-row_B-col_mem.ini`
    Simulation file -> `sst_ip-m_sim_AB.py`


INNER PRODUCT N

    B Matrix (1st matrix) requires:
    - bitmap matrix with col-major format (transposed) -> `B-col_bitmap.in`
    - mem values array with values stored in **col-major** format (same as before)

    A Matrix (2nd matrix) requires:
    - bitmap matrix with col-major format (transposed) -> `A-col_bitmap.in`
    - mem values array with values stored in row-major format (same as before)

    Mem file -> `B-col_A-row_mem.ini`
    Simulation file -> `sst_ip-n_sim_BA.py`

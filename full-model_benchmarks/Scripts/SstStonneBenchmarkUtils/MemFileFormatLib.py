import numpy as np
import struct
import os

"""
Small library with basic functions to work with the mem file format.
Support for read/write CSV files, pack/unpack float values and
conversions between sparse and dense matrices.
"""

def pack_value(value):
    """
    Packs a float value into a 32 bit integer.
    """
    ba = bytearray(struct.pack(">f", value))
    return int.from_bytes(ba, "big")


def unpack_value(value):
    """
    Unpacks a 32 bit integer into a float value.
    """
    ba = value.to_bytes(4, "big")
    return struct.unpack(">f", ba)[0]
    

def csv_read_file(file, unpack_data=True, use_int=False):
    """
    Reads a csv file and returns a list of integers.
    It is used to parse the rowpointer, colpointer and data (mem) files.
    """
    with open(file, "r") as f:
        data = f.read().splitlines()
        data = "".join(data)

        # remove trailing comma if exists
        if data[-1] == ",":
            data = data[:-1]

        data = data.split(",")
        if unpack_data:
            data = [unpack_value(int(x)) for x in data]
        elif use_int:
            data = [int(x) for x in data]
        else:
            data = [float(x) for x in data]

    return data


def csv_write_file(file, data, pack_data=True, use_int=False):
    """
    Writes a csv file from a list of integers.
    It is used to write the rowpointer, colpointer and data (mem) files.
    It appends a trailing comma to the file.
    """
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w") as f:
        if pack_data:
            data = [pack_value(x) for x in data]
            f.write(",".join([str(x) for x in data]) + ",")
        elif use_int:
            f.write(",".join([str(x) for x in data]) + ",")
        else:
            f.write(",".join(['{:.4f}'.format(x) for x in data]) + ",")


def divide_mem(mem_array, bytes_A, bytes_B):
    """
    Splits the mem_array into two arrays of the specified sizes.
    Used to divide mem files into A and B mem arrays
    """
    len_A = bytes_A // 4
    len_B = bytes_B // 4
    assert len(mem_array) == len_A + len_B, f"mem_array length ({len(mem_array)}) does not match the specified lengths ({len_A} + {len_B})"
    return mem_array[:len_A], mem_array[len_A:]


def merge_mem(mem_A, mem_B):
    """
    Merges two mem arrays into a single one.
    Used to merge A and B mem arrays into a single mem array
    """
    return mem_A + mem_B

def matrix_load_from_mem(mem, rows, cols):
    """
    Loads a matrix from the mem file.
    Reads a row-major matrix and returns a numpy matrix.
    """
    assert len(mem) == rows * cols, f"mem length ({len(mem)}) does not match the specified size ({rows * cols})"
    matrix = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            matrix[i, j] = mem[i * cols + j]
    return matrix

def matrix_sparse2dense(rowpointer, colpointer, data, rows, cols):
    """
    Loads a matrix from the rowpointer, colpointer and data (mem).
    Reads a CSR/CSC matrix and returns a numpy matrix.

    row/col pointer MUST be in INT format
    data MUST be unpacked
    """
    assert len(rowpointer) - 1 == rows, f"rowpointer length ({len(rowpointer) - 1}) does not match the specified rows ({rows})"

    matrix = np.zeros((rows, cols))
    data_idx = 0
    for i in range(rows):
        elems_row = colpointer[rowpointer[i]:rowpointer[i+1]]
        for j in elems_row:
            matrix[i, j] = data[data_idx]
            data_idx += 1

    # check if rowpointer and colpointer are correct
    assert data_idx == len(data), f"data_idx ({data_idx}) does not match the specified data length {rowpointer[-1]}"

    return matrix


def matrix_sparsebitmap2dense(bitmap, data, rows, cols):
    """
    Loads a matrix from the bitmap and data (mem).
    Reads a row-major/col-major matrix and returns a numpy matrix.

    bitmap MUST be in INT format
    data MUST be unpacked
    """
    assert len(bitmap) == rows * cols, f"bitmap length ({len(bitmap)}) does not match the specified size ({rows * cols})"

    matrix = np.zeros((rows, cols))
    data_idx = 0
    for i in range(rows):
        for j in range(cols):
            if bitmap[i * cols + j] == 1:
                matrix[i, j] = data[data_idx]
                data_idx += 1

    # check if rowpointer and colpointer are correct
    assert data_idx == len(data), f"data_idx ({data_idx}) does not match the specified data length ({len(data)})"

    return matrix


def matrix_dense2sparse(matrix):
    """
    Stores a matrix to the rowpointer, colpointer and data (mem) files.
    Reads a numpy matrix and creates its CSR/CSC representation.
    Transpose between CSR and CSC has to be done outside this function.
    """
    # generate the new rowpointer and colpointer
    rows, cols = matrix.shape
    new_rowpointer = [0]
    new_colpointer = []
    new_data = []
    for i in range(rows):
        partial_colpointer = []
        for j in range(cols):
            if matrix[i, j] != 0:
                partial_colpointer.append(j)
                value = matrix[i, j]
                new_data.append(value)
        new_colpointer += partial_colpointer
        new_rowpointer.append(new_rowpointer[-1] + len(partial_colpointer))

    return new_rowpointer, new_colpointer, new_data


def matrix_dense2sparsebitmap(matrix):
    """
    Stores a matrix to the bitmap and data (mem) files.
    Reads a numpy matrix and creates its bitmap (row-major) representation.
    If you want to obtain a col-major representation, transpose the matrix before calling this function.
    """
    # generate the new rowpointer and colpointer
    rows, cols = matrix.shape
    new_bitmap = []
    new_data = []
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:
                new_bitmap.append(1)
                value = matrix[i, j]
                new_data.append(value)
            else:
                new_bitmap.append(0)

    return new_bitmap, new_data


def matrix_sparse_mult(A_rowp, A_colp, A_data, A_CSR, rows_A, cols_A, 
                       B_rowp, B_colp, B_data, B_CSR, rows_B, cols_B,
                       result_CSR):
    """
    Performs a sparse matrix multiplication between A and B.
    Returns the resulting rowpointer, colpointer and data (mem) files.
    CSR or CSC result format can be selected.

    data MUST be unpacked
    """
    assert cols_A == rows_B, f"Matrix dimensions do not match ({cols_A} != {rows_B})"
    # load both dense matrices in row-major format
    if A_CSR:
        A = matrix_sparse2dense(A_rowp, A_colp, A_data, rows_A, cols_A)
    else:
        A = matrix_sparse2dense(A_rowp, A_colp, A_data, cols_A, rows_A).T

    if B_CSR:
        B = matrix_sparse2dense(B_rowp, B_colp, B_data, rows_B, cols_B)
    else:
        B = matrix_sparse2dense(B_rowp, B_colp, B_data, cols_B, rows_B).T

    # perform matrix multiplication
    C = A @ B

    # convert result to sparse format
    if not result_CSR:
        C = C.T # CSC
    return matrix_dense2sparse(C)
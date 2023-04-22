#!/bin/bash

# Generate input matrix (note: it is generated in dense format)
python gen_gemm.py

# Start SST simulation
sst sst_stonne_gemm.py
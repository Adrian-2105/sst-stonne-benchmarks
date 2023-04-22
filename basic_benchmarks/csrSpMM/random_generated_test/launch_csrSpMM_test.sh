#!/bin/bash

# Generate input matrix (note: it is generated in dense format)
python gen_csrSpMM.py

# Start SST simulation
sst sst_stonne_csrSpMM.py
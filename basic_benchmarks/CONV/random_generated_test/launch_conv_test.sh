#!/bin/bash

# Generate input matrix (note: it is generated in dense format)
python gen_conv.py

# Start SST simulation
sst sst_stonne_conv.py
#!/bin/bash
#
#SBATCH --workdir=/home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/cluster_outputs_dnn
#SBATCH -J NAME_BENCHMARK
#SBATCH --cpus-per-task=1
#SBATCH --array=1

ID=${SLURM_ARRAY_TASK_ID}

name_benchmark="NAME_BENCHMARK"

#x_id=$(echo "($ID%14) + 1" | bc)
#y_id=$(echo "($ID/14) + 1" | bc)

#outer

cd /home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/$name_benchmark/outer_product


sst /home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/$name_benchmark/outer_product/temporal_sst_stonne_outerProduct.py

cd /home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/$name_benchmark/inner_product

sst /home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/$name_benchmark/inner_product/temporal_sst_stonne_bitmapSpMSpM.py

cd /home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/$name_benchmark/gustavsons

sst /home/users/franciscomm/SST_Simulator/sst-elements/src/sst/elements/sstStonne/tests/scripts/$name_benchmark/gustavsons/temporal_sst_stonne_gustavsons.py




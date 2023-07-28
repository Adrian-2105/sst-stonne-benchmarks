#!/bin/bash

#SBATCH --workdir=/home/adrianfn
#SBATCH -J sst-stonne-randgen-benchmark
#SBATCH --output=randomgen-bench-%A.out
#SBATCH --error=randomgen-bench-%A.err
#SBATCH --exclude=erc[01,02,03,05,06,08,09]

# parameters
TESTS_PER_JOB=5
MODEL_FOLDER=RandGenModelTest
OUTPUT_MODEL_FOLDER="${MODEL_FOLDER}_output" 
SCRIPTS_FOLDER=sst-stonne-benchmarks/full-model_benchmarks/Scripts

# create model folder
mkdir -p $MODEL_FOLDER

# each job will generate a number of tests
for i in $(seq 1 $TESTS_PER_JOB); do
    LAYER_FOLDER=$(printf "$MODEL_FOLDER/randgen_%d%d" $SLURM_JOB_ID $i)
    OUTPUT_FOLDER=$(printf "$OUTPUT_MODEL_FOLDER/randgen_%d%d" $SLURM_JOB_ID $i)
    mkdir -p $LAYER_FOLDER
    mkdir -p $OUTPUT_FOLDER

    echo "Generating random model in $LAYER_FOLDER"
    echo "Output will be in $OUTPUT_FOLDER"

    MIN_SIZE=10
    MAX_SIZE=$(( ($i + 1) * 400 ))
    SPARSITIES=(0 10 20 30 40 50 55 60 65 70 75 80 85 90 95)

    # do while loop to generate random parameters
    M=$(shuf -i $MIN_SIZE-$MAX_SIZE -n 1)
    N=$(shuf -i $MIN_SIZE-$MAX_SIZE -n 1)
    K=$(shuf -i $MIN_SIZE-$MAX_SIZE -n 1)
    # select a random sparsity from the list SPARSITIES
    sparsityA=${SPARSITIES[$(shuf -n 1 -i 0-$(( ${#SPARSITIES[@]} - 1)) )]}
    sparsityB=${SPARSITIES[$(shuf -n 1 -i 0-$(( ${#SPARSITIES[@]} - 1)) )]}
    nnz_A=$((M * K * (100 - sparsityA) / 100))
    nnz_B=$((K * N * (100 - sparsityB) / 100))

    # check that the total number of zeros is less than 10 000 000
    while [ $((nnz_A + nnz_B)) -gt 10000000 ]; do
        # do while loop to generate random parameters
        M=$(shuf -i $MIN_SIZE-$MAX_SIZE -n 1)
        N=$(shuf -i $MIN_SIZE-$MAX_SIZE -n 1)
        K=$(shuf -i $MIN_SIZE-$MAX_SIZE -n 1)
        # select a random sparsity from the list SPARSITIES
        sparsityA=${SPARSITIES[$(shuf -n 1 -i 0-$(( ${#SPARSITIES[@]} - 1)) )]}
        sparsityB=${SPARSITIES[$(shuf -n 1 -i 0-$(( ${#SPARSITIES[@]} - 1)) )]}
        nnz_A=$((M * K * (100 - sparsityA) / 100))
        nnz_B=$((K * N * (100 - sparsityB) / 100))
    done

    echo "Generating random matrix with M=$M, N=$N, K=$K, sparsityA=0.$sparsityA, sparsityB=0.$sparsityB"

    # 1. generate the random matrix
    python3 $SCRIPTS_FOLDER/01_ExtractLayerMatrices/RandomLayerGeneration.py --layer_dir $LAYER_FOLDER --M $M --N $N --K $K --sparsityA 0.$sparsityA --sparsityB 0.$sparsityB --force || exit 1

    # 2. generate the layer environment to execute that layer
    python3 $SCRIPTS_FOLDER/02_EnvironmentGeneration/BuildExecutionEnv.py $LAYER_FOLDER || exit 1

    # 3. execute the layer
    python3 $SCRIPTS_FOLDER/03_Simulations/LaunchSimulation.py --target all --output $OUTPUT_FOLDER $LAYER_FOLDER || { rm -rf "$OUTPUT_FOLDER"; exit 1; }

    # 4. clean the layer environment
    python3 $SCRIPTS_FOLDER/02_EnvironmentGeneration/CleanEnv.py $LAYER_FOLDER || exit 1


    # confirm that the simulation finished correctly
    touch $OUTPUT_FOLDER/ok
done

# after all simulations has been executed, generate the final results MANUALLY
# python3 complete-benchmarks/Scripts/04_Results/GenerateModelResults -c 0  -o "model_statistics_ecfree.csv" Results/$model
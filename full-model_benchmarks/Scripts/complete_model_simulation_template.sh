#!/bin/bash

#SBATCH --workdir={PATH TO BASE DIRECTORY WHERE ALL MODELS ARE FOUND}
#SBATCH -J sst-stonne-benchmark
#SBATCH --cpus-per-task=1
#SBATCH --array=1

# list of models to execute
MODELS=(Alexnet)


for model in $MODELS; do 
    for layer in $(find complete-benchmarks/Models/$model -iname "bench*"); do
        OUTPUT_DIR=Results/$model/$(basename $layer)

        echo "Building environment for layer $layer"
        python complete-benchmarks/Scripts/02_EnvironmentGeneration/BuildExecutionEnv.py $layer
        
        echo "Launching simulation for layer $layer"
        python complete-benchmarks/Scripts/03_Simulations/LaunchSimulation.py --target all --output $OUTPUT_DIR $layer

        echo "Cleaning environment for layer $layer"
        python complete-benchmarks/Scripts/02_EnvironmentGeneration/CleanEnv.py $layer
    done

    echo "Generating results for model $model"
    python complete-benchmarks/Scripts/04_Results/GenerateModelResults -c 0  -o "model_statistics_ecfree.csv" Results/$model
    python complete-benchmarks/Scripts/04_Results/GenerateModelResults -c 10 -o "model_statistics_ec10.csv" Results/$model
done
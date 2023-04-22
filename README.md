# SST-STONNE Benchmarks

This repository contains multiple benchmarks to use and test the SST-STONNE simulator,
from single and basic benchmarks to DNN full-model benchmarks.

Some files are compressed due to GitHub limitations, so please run the following command to
uncompress all files after cloning the repository:

```bash
for file in $(find . -name "*.tar.gz"); do
    tar -xzf $file -C $(dirname $file)
done
```

## Basic Benchmarks

This folder (`basic_benchmarks`) contains some very simple and specific simulation scenarios mainly focused
to test and stress some functionalities of the simulator. To run one of these benchmarks,
just move to the folder and run it with:

```
sst {SST_SIMULATION_SCRIPT}.py
```

## Full-Model Benchmarks

This folder (`full-model_benchmarks`) contains some full-model benchmarks, all of them DNNs.
Each model can be found inside folder `Models`, and each model folder contains a separate folder
for each layer of the model. Inside that folders, you can find a `layer_data` folder which contains
all the necessary data to run all the benchmarks for that layer (all that information has been
generated automatically through scripts and extracting the data from the real DNN models).

### How to run the benchmarks

For more information about each script mentioned below, please take a look to their `--help` menu.

To run these benchmarks properly and take advantage of the scripts provided in this repository,
you need to follow the next steps:

1. Generate your own folder tree structure for the models. If you want to just run the benchmarks
   again (modifying or not the parameters), skip this step. In case you want to generate your own
   environment to simulate your complete model, please take into account the following:
   - Create a folder with `{MODEL_NAME}` as name. Inside that folder, create a separete folder for
     each layer with the name `bench_{MODEL_NAME}_{LAYER_NUM}`. You must follow this naming convention
     or the scripts will not work properly.
   - Create a `bench_{MODEL_NAME}_{LAYER_NUM}/layer_data` folder inside each layer folder. Inside that folder,
     include exactly the same files (preserving the same names) as the examples provided in this repository.
     All of them are mandatory to build the execution environment for the benchmarks.
   - In our case, we used the scripts on `Scripts/01_ExtractLayerMatrices` to generate all the scenarios, but
     they are no longer maintained, so consider only them as a reference.

2. Generate the benchmark environment of each layer. This will generate a different folder for each supported
   dataflow (`inner-product-m`, `inner-product-n`, `outer-product-m`, `outer-product-n`, `gustavsons-m`, `gustavsons-n`).
   Each folder will be a different simulation environment for the layer, and will contain all the necessary files
   to run the simulation. You can generate all the environments at once using the next command:

   ```bash
   for layer_dir in $(find Models/{MODEL_NAME} -type d -name "bench_*"; do
       python Scripts/02_EnvironmentGeneration/BuildExecutionEnvironment.py $layer_dir
    done
   ```

3. Run the benchmarks. Results can be stored in a separate folder, so in my case I prefer to store them
   in a different folder (as reference, you can find my results in the `Results` folder). To run the benchmarks
   and store the results in a different folder, you can use the next command:

   ```bash
    for layer_dir in $(find Models/{MODEL_NAME} -type d -name "bench_*"; do
        python Scripts/03_Simulations/LaunchSimulation.py $layer_dir -o Results/{MODEL_NAME} --target all
    done
    ```

4. Extract the results. Once all dataflows of each layer has been simulated, you can extract its results into a 
   CSV file. This file will contain all the main information of the simulation, including a best mapping-path
   of all the layers of the model, considering Explicit Conversions and without considering them. To can run it with
   the next command, generating in each `Results/{MODEL_NAME}` folder a `model_statistics.csv` file:

   ```bash
    python Scripts/04_Results/GenerateModelResults.py Results/{MODEL_NAME} --ec_cycles 10
    ```

5. (Optional) If you want to check that the result provided by the simulator is correct, use the next command:

    ```bash
    python Scripts/04_Results/GenerateModelResults.py Models/{MODEL_NAME} --results_dir Results/{MODEL_NAME} --delta 0.001
    ```

    _Note:_ in the results included on this repository, output matrices appear with all their values set to 0. This is because
    we considered a modified version of SST-STONNE to simulate them, hiding the write operations and their latency.

### Example of a full simulation script and results

As reference, you can use the `Scripts/complete_model_simulation_template.sh` script, which already includes all the
mentioned commands in the preivous steps and which was used to generate the results of the `Results` folder.

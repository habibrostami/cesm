
### Directory Structure

Before running the data split script, ensure that your directory structure is as follows:

```
project_root/
|
--- data/
|   --- CM/
|   |   --- 0/     # Images labeled as 0 (benign cases)
|   |   --- 1/     # Images labeled as 1 (malignant cases)
|   |
|   --- DM/
|   |   --- 0/     # Corresponding DM images labeled as 0
|   |   --- 1/     # Corresponding DM images labeled as 1
|   |
---packages/
|   ---parameters/
|   |   --- split.py
|   |...
|   ...
|   
--- data_split.py
--- main.py
--- requirements.txt
```

### Running the Data Split Script

1. **Configure the Base Path**:
   - Ensure that the `base_path` and `save_path` in the `data_split.py` script are correctly set to the directory where your data resides.
     By default, both are set to `./data`, meaning the script expects the data to be in a folder named `data` in the project root.

2. **Execute the Script**:
   - Run the script to organize your data:
     ```
     python data_split.py
     ```
   - This script will:
     - Verify that all required images are available (i.e., each CM image has corresponding MLO and DM images).
     - Split the dataset into training, validation, and test sets with a 70-15-15% ratio.
     - Copy the images into the appropriate directories (`train`, `validation`, `test`).
     - Generate three CSV files (`train_data.csv`, `validation_data.csv`, `test_data.csv`) listing the paths to the images and their corresponding labels.

### Model Configuration

1. **Configure the Model in `split.py`**:
   - Open the `split.py` file to configure the models you wish to run. Here s an example configuration:
     ```
     {
         'name': 'kfold-Auto_AOL[cm_cc]',
         'dataset': cm_cc,  # The dataset to be used
         'epochs': 15,      # Number of training epochs
         'resnet_size': 18, # Size of the ResNet backbone (can be 18, 34, 50, etc.)
         'train': False,    # Set this to True to enable training
         'type': 'kfold-simple-run', # Type of model run (refer to main.py for different types)
         'batch_size': 64   # Batch size for training
     }
     ```
   - **Important**: To run a model, set the `'train'` key to `True`. The `'type'` field corresponds to the model type defined in `main.py` under `run_type`.

### Running the Model

1. **Run the Main Script**:
   - Finally, run the `main.py` script to start the training process:
     ```
     python main.py
     ```

## Additional Information

- **Model Types**: Different `run_type` values are available in `main.py`, corresponding to various model architectures and training methods (e.g., `kfold-simple-run`, `joint-unet`, `u-net`, etc.). Adjust the `type` field in `split.py` according to the model you want to train.
- **Results**: Training results, including model checkpoints, will be saved in the specified directories. Check the `save:` path for outputs.


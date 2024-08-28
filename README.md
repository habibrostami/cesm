# How to Run the Code

1. ##Install Dependencies:##
   - Install the required packages using the `cesm_requirements.yaml` file.

2. ##Download the Dataset:##
   - Obtain the dataset from [The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611).

3. ##Organize the Dataset:##
   - Place the CM (Contrast Mammography) and DM (Digital Mammography) images into their respective folders within the `data` directory.
   - Each image type should have `0` and `1` subfolders representing their labels.

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

## 4. Set Up Data Splits: ##
   - Open the `data_split.py` script and verify the `base_path` variable is set to the correct path of your `data` folder.
   - Run `data_split.py` to create the `train`, `validation`, and `test` folders, along with their corresponding `.csv` files (`train.csv`, `validation.csv`, `test.csv`).

## 5. Configure the Model and Dataset: ##
   - Open `packages/parameters/split.py` and set the desired model and dataset to `True` (e.g., `cm_cc` on U-Net).
   - Supported models are defined in `main.py`. Any model set to `True` in `split.py` will be run when executing `main.py`.

## 6. Run the Model##
   - Execute `main.py` to start the training process.

## Model Configuration

### Configure the Model in `split.py` ###:
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


## Additional Information

- **Model Types**: Different `run_type` values are available in `main.py`, corresponding to various model architectures and training methods (e.g., `kfold-simple-run`, `joint-unet`, `u-net`, etc.). Adjust the `type` field in `split.py` according to the model you want to train.
- **Results**: Training results, including model checkpoints, will be saved in the specified directories. Check the `save:` path for outputs.

# Datasets Citation
We would like to express our gratitude to Khaled R. et al. and the TCIA repository for publishing and making the CESM dataset publicly available, which served as the foundation for our work. We also extend our thanks to Moreira et al. for providing the INbreast dataset, which we used as an external test set.

- Khaled R., Helal M., Alfarghaly O., Mokhtar O., Elkorany A., El Kassas H., Fahmy A. Categorized Digital Database for Low Energy and Subtracted Contrast Enhanced Spectral Mammography Images [Dataset]. (2021) The Cancer Imaging Archive. DOI: [10.7937/29kw-ae92](https://doi.org/10.7937/29kw-ae92)

- Khaled, R., Helal, M., Alfarghaly, O., Mokhtar, O., Elkorany, A., El Kassas, H., & Fahmy, A. Categorized Contrast Enhanced Mammography Dataset for Diagnostic and Artificial Intelligence Research. (2022) Scientific Data, Volume 9, Issue 1. DOI: [10.1038/s41597-022-01238-0](https://doi.org/10.1038/s41597-022-01238-0)

- Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging, Volume 26, Number 6, December 2013, pp 1045-1057. DOI: [10.1007/s10278-013-9622-7](https://doi.org/10.1007/s10278-013-9622-7)

- Inês C. Moreira, Igor Amaral, Inês Domingues, António Cardoso, Maria João Cardoso, Jaime S. Cardoso, INbreast: Toward a Full-field Digital Mammographic Database, Academic Radiology, Volume 19, Issue 2, 2012, Pages 236-248, ISSN 1076-6332, DOI: [10.1016/j.acra.2011.09.014](https://doi.org/10.1016/j.acra.2011.09.014). [Link to Article](https://www.sciencedirect.com/science/article/pii/S107663321100451X)

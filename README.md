# DOPE

This repo contains scripts for DOPE model in *paper name*. The data acquisition methods as well as model architecture details are also described below.


## Environment Setup

We recommend using `conda` to setup the required python environment for the project.

```bash
conda create -n dope python=3.7
conda install -c anaconda jupyter
conda install -c anaconda pandas
```

As the dataset provided will exceed the limitation of Github file size, we will use [Git Large File Storage](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Please follow the instructions to install it so everything can run successully.


## Data

### Biomed-clinical data


### Multi-omics data



We are using multi-omics data for training our model. 

preprocess.py includes functions that convert the raw text files holding biomedical and clinical data into csvs through pandas dataframes. The first column contains the patients' uuids ('bcr_patient_uuid') and the subsequent columns include features, such as 'pharmaceutical_therapy_type', 'retrospective_collection', 'gender', etc. Some patient uuids of the biomedical data do not correspond to select information collected for the clinical data so those entries currently are NaN, which we will determine how to further process for training.

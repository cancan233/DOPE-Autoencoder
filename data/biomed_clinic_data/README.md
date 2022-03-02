# Biomedical and Clinical Datasets


> 🚧 The patients in biomed-clinical data is represented by bcr_patient_uuid

## Download dataset and Dataset details

* [Biomedical] (Available at https://wiki.cancerimagingarchive.net/display/Public/TCGA-OV#75694970aa49cd675604c35a9d171bde3194990). Size: (33138, 112)

* [Clinical] (Available at https://wiki.cancerimagingarchive.net/display/Public/TCGA-OV#75694970aa49cd675604c35a9d171bde3194990). Size: (4294, 161)
    

## Dataset Preprocess details

For the biomedical and clinical data, each `.txt` file was converted into a `.csv`. 

```bash
python preprocess.py
```

The biomedical dataframes were then concatenated with the clinical dataframes in `biomed_data_preparation.ipynb`, yielding a size = (234454, 272) dataframe called `biomedical_clinical_data.csv`.
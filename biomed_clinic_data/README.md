# Biomedical and Clinical Datasets

## Download dataset
The jupyter notebook `omics_data_preparation.ipynb` provides way to download the four types omics data and proprocess procedure. 


## Dataset details

* [Biomedical] (Available at https://wiki.cancerimagingarchive.net/display/Public/TCGA-OV#75694970aa49cd675604c35a9d171bde3194990). Size: (25249, 117)

* [Clinical] (Available at https://wiki.cancerimagingarchive.net/display/Public/TCGA-OV#75694970aa49cd675604c35a9d171bde3194990). Size: (4347, 131)
    

## Dataset Preprocess details

For the biomedical and clinical data, each .txt file was converted into a .csv and then a pandas dataframe. The biomedical dataframes were then concatenated with the clinical dataframes, yielding a size = (29596, 244) dataframe called biomedical_clinical_data.csv in the biomed_clinic_data folder.



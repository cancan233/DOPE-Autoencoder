## Requirements

> 🚧 The patients in biomed-clinical data is represented by bcr_patient_uuid
> 
> 🚧 The patients in multi-omics data is represented by bcr_sample_barcode

## Usage

Use `python run.py --help` to check all available arguments. All hyperparameters are included in `hyperparameters.py`.

### Train autoencoder for omics data
```bash
python run.py --autoencoder-model vanilla \
              --omics-data ../omics_data/cnv_methyl_rnaseq.csv \
              --train-autoencoder
```

### Train classifier for encoded omics data and biomed data

```bash
python run.py --autoencoder-model vanilla \
              --classifier-model all \
              --load-autoencoder ./output/checkpoints/121021-203852/epoch_19 \
              --biomed-data ./data/biomed.csv \
              --merged-data ./data/cnv_methyl_rnaseq_biomed.csv \
              --train-classifier \
              --classifier-data merged \
              --no-save
```

### Train classifier for only omics data

```bash
python run.py --autoencoder-model vanilla \
              --classifier-model all \
              --load-autoencoder ./output/checkpoints/121021-203852/epoch_19 \
              --biomed-data ./data/biomed.csv \
              --merged-data ./data/cnv_methyl_rnaseq_biomed.csv \
              --train-classifier \
              --classifier-data omics \
              --no-save
```

### Train classifier for only biomed data

```bash
python run.py --classifier-model all \
              --biomed-data ./data/biomed.csv \
              --merged-data ./data/cnv_methyl_rnaseq_biomed.csv \
              --train-classifier \
              --classifier-data biomed \
              --no-save
```

## Workflow for producing dataframe contains the encoded omics features together with the biomed-clinical features

1. Generate the encoded omics features, i.e., latent features produced by the autoencoder. Go into the folder `encoded_omics_data`, and follow the commands in `Encoding_omics_features_with_autoencoder.ipynb` to generate the encoded omics data. 

2. Open `merge_omics_biomed_clinical_data.ipyne`, go section that combine the encoded omics features with the biomed-clincial features. Please read through the command arguments, change them if necessary, especially checkpoint directory, tri-omics data path, etc.

## Reference
1. VAE: https://gist.github.com/RomanSteinberg/c4a47470ab1c06b0c45fa92d07afe2e3
## Requirements

* python 3.6
* tensorflow 2.x

> 🚧 The biomed data is not indexed by uuid, which is index by number 0 instead. The uuid column is at the end of the data table.
> 
> 🚧 The omics data and the merged data is indexed by barcode. 


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

## Results

## Reference
1. VAE: https://gist.github.com/RomanSteinberg/c4a47470ab1c06b0c45fa92d07afe2e3
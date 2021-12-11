## Requirements

* python 3.6
* tensorflow 2.x

## Usage

Use `python run.py --help` to check all available arguments. All hyperparameters are included in `hyperparameters.py`.

### Train autoencoder for omics data
```bash
python run.py --autoencoder-model vanilla \
              --omics-data ../omics_data/cnv_methyl_rnaseq.csv \
              --train-autoencoder
```

### Train classifier for encoded omics data and clinical data

```bash
python run.py --classifier-model all \
              --load-autoencoder ./output/checkpoints/121021-203852/ \
              --merged-data ./data/cnv_methyl_rnaseq_clinical.csv \
              --train-classifier merged
```

### Train classifier for only clinical data

```bash
python run.py --classifier-model all \
              --biomed-data ./data/clinical.csv \
              --train-classifier biomed
```

## Reference
1. VAE: https://gist.github.com/RomanSteinberg/c4a47470ab1c06b0c45fa92d07afe2e3
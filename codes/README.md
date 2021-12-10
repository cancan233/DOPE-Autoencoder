## Usage

Use `python run.py --help` to check all available arguments. All hyperparameters are included in `hyperparameters.py`.

### Train
```bash
python run.py --autoencoder-model vanilla_autoencoder\
              --omics-data ../omics_data/cnv_methyl_rnaseq.csv\
              --merged-data ./merged.csv --no-save --predict
```

<!-- ### Test
```bash
python run.py --model vanilla_autoencoder\
              --evaluate \
``` -->

## Reference
1. VAE: https://gist.github.com/RomanSteinberg/c4a47470ab1c06b0c45fa92d07afe2e3
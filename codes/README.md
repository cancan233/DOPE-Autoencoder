## Usage

Use `python run.py --help` to check all available arguments. All hyperparameters are included in `hyperparameters.py`.

### Train
```bash
python run.py --model vanilla_autoencoder\
              --omics-data ../omics_data/cnv_methyl_rnaseq.csv\
```

### Test
```bash
python run.py --model vanilla_autoencoder\
              --evaluate \
```

## Reference

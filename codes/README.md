## Usage

Use `python run.py --help` to check all available arguments. All hyperparameters are included in `hyperparameters.py`.

### Train
```bash
python run.py --model vanilla_autoencoder\
              --omics-data ../omics_data/cnv_methyl_rnaseq.csv\
              --merged-data ./merged.csv --no-save --predict
```

### Test
```bash
python run.py --model vanilla_autoencoder\
              --evaluate \
```

## Reference
1. https://gist.github.com/RomanSteinberg/c4a47470ab1c06b0c45fa92d07afe2e3

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-1x1wxZGXCiKVuInRJu4HpdZG62GcB27' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-1x1wxZGXCiKVuInRJu4HpdZG62GcB27" -O cnv_methyl_mrna.csv && rm -rf /tmp/cookies.txt
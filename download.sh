#!/bin/bash
# Cancan Huang @ Brown University

cd ./data/omics_data
mkdir 1_csv_data
cd 1_csv_data

FILEID="1-2q5xKyCBwZjwUdv1KwlSpCUJQ19vk9l"
FILENAME='cnv_methyl_rnaseq.csv'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt

FILEID="1-1x1wxZGXCiKVuInRJu4HpdZG62GcB27"
FILENAME='cnv_methyl_mrna.csv'
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt


# cd ./dope
# FILEID="1NZHJ2-IWBSysH8mr1cTXlmTmWS0c0WGm"
# FILENAME='cnv_methyl_mrna_biomed_clinical_85_features.csv'
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
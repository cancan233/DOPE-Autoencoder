import os
import glob
import csv
import pandas as pd
import numpy as np

def save_as_csv():
    path = './raw_data/gdc_download_biomedical_ov/'
    read_aliquot = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_aliquot_ov.txt')
    read_aliquot.to_csv (r'./csv_data/aliquot.csv', index=None)

    read_analyte = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_analyte_ov.txt')
    read_analyte.to_csv (r'./csv_data/analyte.csv', index=None)

    read_diag_slides = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_diagnostic_slides_ov.txt')
    read_diag_slides.to_csv (r'./csv_data/diag_slides.csv', index=None)

    read_portion = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_portion_ov.txt')
    read_portion.to_csv (r'./csv_data/portion.csv', index=None)

    read_protocol = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_protocol_ov.txt')
    read_protocol.to_csv (r'./csv_data/protocol.csv', index=None)

    read_sample = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_sample_ov.txt')
    read_sample.to_csv (r'./csv_data/sample.csv', index=None)

    read_shipment_portion = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_shipment_portion_ov.txt')
    read_shipment_portion.to_csv (r'./csv_data/shipment_portion.csv', index=None)

    read_slide = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_slide_ov.txt')
    read_slide.to_csv (r'./csv_data/slide.csv', index=None)

    read_ssf_norm = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_ssf_normal_controls_ov.txt', error_bad_lines=False)
    read_ssf_norm.to_csv (r'./csv_data/ssf_norm.csv', index=None)

    read_ssf_tumor = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_ssf_tumor_samples_ov.txt', error_bad_lines=False)
    read_ssf_tumor.to_csv (r'./csv_data/ssf_tumor.csv', index=None)


def merge_csv():
    path = "./csv_data/"

    all_files = glob.glob(os.path.join(path, "*.csv"))

    all_df = []
    for f in all_files:
        df = pd.read_csv(f, sep=',')
        all_df.append(df)

    merged_df = pd.concat(all_df, ignore_index=True, sort=True)
    merged_df.to_csv (r'./biomed_data.csv', index=None)

def main():
    save_as_csv()
    merge_csv()

if __name__ == "__main__":
    main()
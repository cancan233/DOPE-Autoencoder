import os
import glob
import csv
import pandas as pd
import numpy as np


def save_as_csv():
    """
    Saves all raw txt files as separate csvs to later merge.
    """

    # Biomedical data:
    read_aliquot = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_aliquot_ov.txt",
        sep="\t",
    )
    read_aliquot.drop(0, 0, inplace=True)
    read_aliquot.to_csv(r"./01_csv_data/biomed_aliquot.csv", index=None)

    no_duplicates_df = read_aliquot.drop_duplicates(subset=["bcr_patient_uuid"])

    no_duplicates_df.to_csv("./01_csv_data/biomed_aliquot_short.csv", index=None)

    read_analyte = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_analyte_ov.txt",
        sep="\t",
    )
    read_analyte.drop(0, 0, inplace=True)
    read_analyte.to_csv(r"./01_csv_data/biomed_analyte.csv", index=None)

    read_diag_slides = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_diagnostic_slides_ov.txt",
        sep="\t",
    )
    read_diag_slides.drop(0, 0, inplace=True)
    read_diag_slides.to_csv(r"./01_csv_data/biomed_diag_slides.csv", index=None)

    read_portion = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_portion_ov.txt",
        sep="\t",
    )
    read_portion.drop(0, 0, inplace=True)
    read_portion.to_csv(r"./01_csv_data/biomed_portion.csv", index=None)

    read_protocol = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_protocol_ov.txt",
        sep="\t",
    )
    read_protocol.drop(0, 0, inplace=True)
    read_protocol.to_csv(r"./01_csv_data/biomed_protocol.csv", index=None)

    read_sample = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_sample_ov.txt",
        sep="\t",
    )
    read_sample.drop(0, 0, inplace=True)
    read_sample.to_csv(r"./01_csv_data/biomed_sample.csv", index=None)

    read_shipment_portion = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_shipment_portion_ov.txt",
        sep="\t",
    )
    read_shipment_portion.drop(0, 0, inplace=True)
    read_shipment_portion.to_csv(
        r"./01_csv_data/biomed_shipment_portion.csv", index=None
    )

    read_slide = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_slide_ov.txt",
        sep="\t",
    )
    read_slide.drop(0, 0, inplace=True)
    read_slide.to_csv(r"./01_csv_data/biomed_slide.csv", index=None)

    read_ssf_norm = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_ssf_normal_controls_ov.txt",
        sep="\t",
    )
    read_ssf_norm.drop([0, 1], 0, inplace=True)
    read_ssf_norm.to_csv(r"./01_csv_data/biomed_ssf_norm.csv", index=None)

    read_ssf_tumor = pd.read_csv(
        r"./00_raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_ssf_tumor_samples_ov.txt",
        sep="\t",
    )
    read_ssf_tumor.drop([0, 1], 0, inplace=True)
    read_ssf_tumor.to_csv(r"./01_csv_data/biomed_ssf_tumor.csv", index=None)

    # Clinical data:
    read_drug = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_drug_ov.txt",
        sep="\t",
    )
    read_drug.drop([0, 1], 0, inplace=True)
    read_drug.to_csv(r"./01_csv_data/clinical_drug.csv", index=None)

    read_v1_nte = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_follow_up_v1.0_nte_ov.txt",
        sep="\t",
    )
    read_v1_nte.drop([0, 1], 0, inplace=True)
    read_v1_nte.to_csv(r"./01_csv_data/clinical_v1_nte.csv", index=None)

    read_v1 = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_follow_up_v1.0_ov.txt",
        sep="\t",
    )
    read_v1.drop([0, 1], 0, inplace=True)
    read_v1.to_csv(r"./01_csv_data/clinical_v1.csv", index=None)

    read_nte = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_nte_ov.txt",
        sep="\t",
    )
    read_nte.drop([0, 1], 0, inplace=True)
    read_nte.to_csv(r"./01_csv_data/clinical_nte.csv", index=None)

    read_omf_v4 = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_omf_v4.0_ov.txt",
        sep="\t",
    )
    read_omf_v4.drop([0, 1], 0, inplace=True)
    read_omf_v4.to_csv(r"./01_csv_data/clinical_omf_v4.csv", index=None)

    read_patient = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_patient_ov.txt",
        sep="\t",
    )
    read_patient.drop([0, 1], 0, inplace=True)
    read_patient.to_csv(r"./01_csv_data/clinical_patient.csv", index=None)

    read_radiation = pd.read_csv(
        r"./00_raw_data/gdc_download_clinical_ov/nationwidechildrens.org_clinical_radiation_ov.txt",
        sep="\t",
    )
    read_radiation.drop([0, 1], 0, inplace=True)
    read_radiation.to_csv(r"./01_csv_data/clinical_radiation.csv", index=None)


def merge_csv():
    """
    Concatenates all csvs into a single merged_bioclin_data.csv
    """

    path = "./01_csv_data"

    all_files = glob.glob(os.path.join(path, "*.csv"))

    all_df = []
    for f in all_files:
        df = pd.read_csv(f, sep=",")
        all_df.append(df)

    merged_df = pd.concat(all_df, ignore_index=True, sort=True)
    merged_df.to_csv(r"./merged_bioclin_data.csv", index=None)


def main():
    save_as_csv()
    # merge_csv()


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np


def merge_omics_clinical(omics_df, clinical_df):
    (num_patients, num_features) = omic_df.shape
    print(
        f"Number of features: {num_features} \t Number of patients: {num_patients} in omics data"
    )

    (num_patients, num_features) = clinical_df.shape
    print(
        f"Number of features: {num_features} \t Number of patients: {num_patients} in clinical data"
    )

    # change uuid to barcode
    clinical_df["bcr_patient_barcode"] = "TEST"
    for i in range(clinical_df.shape[0]):
        if clinical_df["bcr_patient_uuid"][i] in uuid2barcode:
            clinical_df["bcr_patient_barcode"][i] = (
                uuid2barcode[clinical_df["bcr_patient_uuid"][i]] + "-01"
            )
        else:
            clinical_df["bcr_patient_barcode"][i] = None

    # Temporary only take the "treatment_outcome_first_course_x" data
    target_df = clinical_df[["treatment_outcome_first_course_x", "bcr_patient_barcode"]]
    target_df = target_df.drop_duplicates()
    target_df = target_df.set_index("bcr_patient_barcode")

    merged_df = pd.merge(omic_df, target_df, left_index=True, right_index=True)
    merged_df.to_csv("merged.csv")


def dict_uuid2barcode():
    uuid2barcode = {}
    path = "../csv_data"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    for f in all_files:
        df = pd.read_csv(f, sep=",")
        if "bcr_patient_uuid" in df and "bcr_patient_barcode" in df:
            for i in range(2, df.shape[0]):
                if df["bcr_patient_uuid"][i] not in uuid2barcode:
                    uuid2barcode[df["bcr_patient_uuid"][i]] = df["bcr_patient_barcode"][
                        i
                    ]
    return uuid2barcode


def main():
    omic_df = pd.read_csv("cnv_methyl_rnaseq.csv", index_col=0).T
    omic_df = omic_df.astype("float32")
    clinical_df = (
        pd.read_csv("clinical.csv", index_col=0).drop_duplicates().reset_index()
    )
    uuid2barcode = dict_uuid2barcode()
    merge_omics_clinical(omics_df, clinical_df, uuid2barcode)


if __name__ == "__main__":
    main()

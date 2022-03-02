import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import json

# merge(), uuid2barcode(), reindex()


def merge_omics_biomed(omic_df, biomed_df, uuid2barcode, merged_name):
    (num_patients, num_features) = omic_df.shape
    print(
        f"Number of features: {num_features} \t Number of patients: {num_patients} in omics data"
    )

    (num_patients, num_features) = biomed_df.shape
    print(
        f"Number of features: {num_features} \t Number of patients: {num_patients} in biomed data"
    )

    # move "treatment_outcome_first_course_x" to the last column
    temp_cols = biomed_df.columns.tolist()
    index = biomed_df.columns.get_loc("treatment_outcome_first_course_x")
    new_cols = (
        temp_cols[0:index] + temp_cols[index + 1 :] + temp_cols[index : index + 1]
    )
    biomed_df = biomed_df[new_cols]

    # change uuid to barcode
    biomed_df["bcr_patient_barcode"] = "TEST"
    for i in range(biomed_df.shape[0]):
        if biomed_df["bcr_patient_uuid"][i] in uuid2barcode:
            biomed_df["bcr_patient_barcode"][i] = uuid2barcode[
                biomed_df["bcr_patient_uuid"][i]
            ]
        else:
            biomed_df["bcr_patient_barcode"][i] = None
    biomed_df.drop(columns=["bcr_patient_uuid"], inplace=True)

    # Temporary only take the "treatment_outcome_first_course_x" data
    target_df = biomed_df
    target_df = target_df.drop_duplicates()
    target_df = target_df.set_index("bcr_patient_barcode")

    merged_df = pd.merge(omic_df, target_df, left_index=True, right_index=True)
    merged_df.to_csv("./data/{}".format(merged_name))


def dict_uuid2barcode():
    uuid2barcode = {}
    path = "../biomed_clinic_data/csv_data"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    for f in all_files:
        df = pd.read_csv(f, sep=",")
        if "bcr_patient_uuid" in df and "bcr_patient_barcode" in df:
            for i in range(2, df.shape[0]):
                if df["bcr_patient_uuid"][i] not in uuid2barcode:
                    uuid2barcode[df["bcr_patient_uuid"][i]] = (
                        df["bcr_patient_barcode"][i] + "-01"
                    )
    return uuid2barcode


def merge():
    omic = "cnv_methyl_mrna"
    # omic = "cnv_methyl_rnaseq"
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    biomed = "biomed_85features"

    merged_name = "{}_{}_{}.csv".format(omic, biomed, timestamp)
    print("merging {} and biomed data at {}".format(omic, merged_name))
    omic_df = pd.read_csv("../omics_data/{}.csv".format(omic), index_col=0).T
    omic_df = omic_df.astype("float32")
    biomed_df = (
        pd.read_csv("./data/{}.csv".format(biomed), index_col=0,)
        .drop_duplicates()
        .reset_index()
    )
    with open("uuid2barcode.json", "r") as file:
        uuid2barcode = json.load(file)
    merge_omics_biomed(omic_df, biomed_df, uuid2barcode, merged_name)


def save_uuid2barcode():
    uuid2barcode = dict_uuid2barcode()

    with open("uuid2barcode.json", "w") as file:
        json.dump(uuid2barcode, file)

    with open("uuid2barcode.json", "r") as file:
        new_dict = json.load(file)

    print(new_dict)


if __name__ == "__main__":
    merge()
    
    # uuid2barcode()

    reindex()

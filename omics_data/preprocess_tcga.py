import pandas as pd
import numpy as np
import os


def min_max_normalization(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )
    return df


def remove_missing_zero_data(df):
    df = df.dropna(axis=0, how="any")
    df = df[df.where(df != 0).any(axis=1)]
    return df


def load_raw_data(path):
    data_cnv_file = "CNV"
    data_mRNA_file = "mRNA"
    data_RNAseq_file = "RNAseq"
    data_methyl_file = "Methyl"
    data_gdc_methyl_file = "gdc_methyl"

    df_list["cnv"] = pd.read_csv(path + os.sep + data_cnv_file, sep="\t", index_col=0)
    df_list["mrna"] = pd.read_csv(path + os.sep + data_mRNA_file, sep="\t", index_col=0)
    df_list["rnaseq"] = pd.read_csv(
        path + os.sep + data_RNAseq_file, sep="\t", index_col=0
    )
    df_list["methyl"] = pd.read_csv(
        path + os.sep + data_methyl_file, sep="\t", index_col=0
    )
    df_list["gdc_methyl"] = pd.read_csv(
        path + os.sep + data_gdc_methyl_file, sep="\t", index_col=0
    )

    for key in df_list.keys():
        print(f"{key} \t {df_list[key].shape}")

    return df_list


def tri_omics_inter(df1, df2, df3):
    col_idx = pd.concat([df1, df2, df3], axis=0, join="inner").columns
    return col_idx


def main():
    print("===== loading raw dataset =====")
    df_list = load_raw_data("./tmp")

    print("===== CNV + mRNA + DNA methylation =====")
    col_idx = tri_omics_inter(df_list["cnv"], df_list["mrna"], df_list["methyl"])
    print(len(col_idx))

    df_list["cnv_1"] = df_list["cnv"][col_idx]
    df_list["mrna_1"] = df_list["mrna"][col_idx]
    df_list["methyl_1"] = df_list["methyl"][col_idx]

    print("===== CNV + RNAseq + DNA methylation =====")
    col_idx = tri_omics_inter(df_list["cnv"], df_list["rnaseq"], df_list["methyl"])

    col_idx = pd.concat(
        [df_list["cnv"], df_list["rnaseq"], df_list["methyl"]], axis=0, join="inner"
    ).columns
    print(len(col_idx))

    df_list["cnv_2"] = df_list["cnv"][col_idx]
    df_list["rnaseq_2"] = df_list["rnaseq"][col_idx]
    df_list["methyl_2"] = df_list["methyl"][col_idx]

    for key in df_list.keys():
        df_list[key] = remove_missing_zero_data(df_list[key])
        print(f"{key} \t {df_list[key].shape}")
    for key in df_list.keys():
        df_list[key] = min_max_normalization(df_list[key])

    final_list = {}

    final_list["mrna_methyl"] = pd.concat(
        [df_list["mrna_1"], df_list["methyl_1"]], axis=0, join="inner"
    )
    final_list["methyl_cnv"] = pd.concat(
        [df_list["cnv_1"], df_list["methyl_1"]], axis=0, join="inner"
    )
    final_list["cnv_mrna"] = pd.concat(
        [df_list["cnv_1"], df_list["mrna_1"]], axis=0, join="inner"
    )

    final_list["methyl_rnaseq"] = pd.concat(
        [df_list["methyl_2"], df_list["rnaseq_2"]], axis=0, join="inner"
    )
    final_list["cnv_rnaseq"] = pd.concat(
        [df_list["cnv_2"], df_list["rnaseq_2"]], axis=0, join="inner"
    )

    final_list["cnv_methyl_mrna"] = pd.concat(
        [final_list["methyl_cnv"], df_list["mrna_1"]], axis=0, join="inner"
    )
    final_list["cnv_methyl_rnaseq"] = pd.concat(
        [final_list["methyl_rnaseq"], df_list["cnv_2"]], axis=0, join="inner"
    )

    for key in final_list.keys():
        print(f"{key} \t {final_list[key].shape}")

    for key in final_list.keys():
        final_list[key].to_csv(f"{key}.csv")
    for key in df_list.keys():
        df_list[key].to_csv(f"{key}.csv")


if __name__ == "__main__":
    main()

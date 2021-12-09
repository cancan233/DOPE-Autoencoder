import pandas as pd
import numpy as np


def remove_missing_zero_data(df):
    df = df.dropna(axis=0, how="any")
    df = df[df.where(df != 0).any(axis=1)]
    return df


def min_max_normalization(df):
    for column in df.columns:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )
    return df


def main():
    df_list = {}
    path = "./tmp/"

    data_cnv_file = "cnv.tsv"
    data_RNAseq_file = "rnaseq.tsv"
    data_methyl_file = "dnamethyl.tsv"

    df_list["cnv"] = pd.read_csv(path + data_cnv_file, sep="\t", index_col=0)
    df_list["rnaseq"] = pd.read_csv(path + data_RNAseq_file, sep="\t", index_col=0)
    df_list["methyl"] = pd.read_csv(path + data_methyl_file, sep="\t", index_col=0)

    for key in df_list.keys():
        print(f"{key} \t {df_list[key].shape}")

    col_idx = pd.concat(
        [df_list["cnv"], df_list["rnaseq"], df_list["methyl"]], axis=0, join="inner"
    ).columns
    print(len(col_idx))

    df_list["cnv"] = df_list["cnv"][col_idx]
    df_list["rnaseq"] = df_list["rnaseq"][col_idx]
    df_list["methyl"] = df_list["methyl"][col_idx]

    for key in df_list.keys():
        df_list[key] = remove_missing_zero_data(df_list[key])
        print(f"{key} \t {df_list[key].shape}")

    for key in df_list.keys():
        df_list[key] = min_max_normalization(df_list[key])

    final_list = {}
    final_list["methyl_rnaseq"] = pd.concat(
        [df_list["methyl"], df_list["rnaseq"]], axis=0, join="inner"
    )

    final_list["cnv_methyl_rnaseq"] = pd.concat(
        [final_list["methyl_rnaseq"], df_list["cnv"]], axis=0, join="inner"
    )

    for key in final_list.keys():
        print(f"{key} \t {final_list[key].shape}")
    for key in final_list.keys():
        final_list[key].to_csv(f"{key}_gdc.csv")


if __name__ == "__main__":
    main()

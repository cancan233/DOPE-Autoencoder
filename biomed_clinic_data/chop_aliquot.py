import pandas as pd

df = pd.read_csv (r'./raw_data/gdc_download_biomedical_ov/nationwidechildrens.org_biospecimen_aliquot_ov.txt', sep='\t')
df.drop(0, 0, inplace=True)

no_duplicates_df = df.drop_duplicates(subset=['bcr_patient_uuid'])

no_duplicates_df.to_csv("./csv_data/aliquot_short.csv", index=None)

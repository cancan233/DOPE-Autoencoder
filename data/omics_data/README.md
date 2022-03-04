# Omics dataset TCGA


> 🚧 The patients in multi-omics data is represented by bcr_sample_barcode

## Download dataset
The jupyter notebook `omics_data_preparation.ipynb` provides way to download the four types omics data and proprocess procedure. We have provided the after-preproccessed csv files in [Google Drive](https://drive.google.com/drive/folders/1-I54hMQOTHLTsKpf26pe_yDyIp-2HoB5?usp=sharing) [The sizes of files exceed the Github LFS limitation.]

You can run `download.sh` to download the two preprocessed tri-omics data files: `cnv_methyl_rnaseq.csv` and `cnv_methyl_mrna.csv`.

## Dataset summary

| Type | File Name | (num_features, num_patients) |
|------|------|-------|
| cnv | cnv.csv | (24776, 579) |
| mrna | mrna.csv | (12042, 593) |
| rnaseq | rnaseq.csv | (20530, 308) |
| methyl | methyl.csv | (27578, 616) |
| mrna_methyl | mrna_methyl.csv | (33736, 555) |
| methyl_cnv | methyl_cnv.csv | (46470, 555) |
| cnv_mrna | cnv_mrna.csv | (36818, 555) |
| methyl_rnaseq | methyl_rnaseq.csv | (42846, 292) |
| cnv_rnaseq | cnv_rnaseq.csv | (45094, 292) |
| cnv_methyl_mrna | cnv_methyl_mrna.csv | (58512, 555) |
| cnv_methyl_rnaseq | cnv_methyl_rnaseq.csv | (67622, 292) |

In google drive folder, cnv_1, mrna_1 and methyl_1 is sub-dataset for cnv_methyl_mrna. cnv_2, rnaseq_2 and methyl_2 is sub-dataset for cnv_methyl_rnaseq

## Dataset details

The notebook will download four types of data: CNV, mRNA, DNA methylation, RNAseq. All data are downloaded from the TCGA Ovarian Cancer (OV) cohort hosted in [UCSC Xena data portal](https://xenabrowser.net/datapages/). All size are specified in the format (num_features, num_patients)

* [CNV](https://xenabrowser.net/datapages/?dataset=TCGA.OV.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443): copy number variation (gene-level). Size: (24776, 579).
    * Copy number profile was measured experimentally using whole genome microarry. Subsequently, TCGA FIREHOSE pipeline applied GISTIC2 method to produce segmented CNV data, which was then mapped to genes to produce gene-level estimates. 

* [mRNA](https://xenabrowser.net/datapages/?dataset=TCGA.OV.sampleMap%2FHT_HG-U133A&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443). Size: (12042, 593)
    * The gene expression profile was measured experimentally using the Affymetrix HT Human Genome U133a mircoarrayu platform. This dataset shows the gene-level transcription estimates. Data is in log space. Each gene or exon has been centered to zero by default. 

* [RNAseq](https://xenabrowser.net/datapages/?dataset=TCGA.OV.sampleMap%2FHiSeqV2_exon&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443): exon espression RNAseq. Size: (20530, 308)
    * The exon expression profile was measured experimentally using the Illumina HiSeq 2000 RNA Sequencing platform. This dataset shows the exon-level transcription estimates, as in RPKM values (Reads Per Kilobase of exon model per Million mapped reads). Each gene or exon has been centered to zero by default. 

* [DNA methylation](https://xenabrowser.net/datapages/?dataset=TCGA.OV.sampleMap%2FHumanMethylation27&host=https%3A%2F%2Ftcga.xenahubs.net&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443). Size: (27578, 616)
    * DNA methylation profile was measured experimentally using the Illumina Infinium HumanMethylation27 platform. DNA methylation beta values are continuous variable between 0 and 1, representing the ratio of the intensity of the methylated bead type to the combined locus intensity. Higher beta values represent higher level of DNA methylation. 

## Dataset Preprocess details
The data preprocess follows the steps same as [Hira et al.](https://www.nature.com/articles/s41598-021-85285-4.pdf)

<figure>
    <img src="./figures/summary_table.png" alt="summary_data">
    <figcaption>Summary of data combinations included in the dataset folders</figcaption>
</figure>   

<figure>
    <img src="./figures/preprocess_flowchart.png " alt="summary_data">
    <figcaption>Data preprocessing with di- and tri-omics data generation. The figure is taken from Hira's paper. Thus the feature number and sample number might be different.</figcaption>
</figure>
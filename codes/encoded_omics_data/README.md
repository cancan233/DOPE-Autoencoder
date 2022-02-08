# README

This folder includes the encoded omics features after training autoencoder model with following parameters:
``` bash
num_epochs = 50
learning_rate = 1e-4
epsilon = 1e-5
batch_size = 4
intermediate_dim = 2048
latent_dim = 128
```

Following files are included:
- `vanillaAE_cnv_methyl_mrna_biomed_85features.csv`: encoded omics features for the merged dataset between cnv_methyl_mrna and biomed_85features
- `vanillaAE_cnv_methyl_rnaseq_biomed_85features.csv`: encoded omics features for the merged dataset between cnv_methyl_rnaseq and biomed_85features
- `variationalAE_cnv_methyl_mrna_biomed_85features.csv`: encoded omics features for the merged dataset between cnv_methyl_mrna and biomed_85features
- `variationalAE_cnv_methyl_rnaseq_biomed_85features.csv`: encoded omics features for the merged dataset between cnv_methyl_rnaseq and biomed_85features


To obtain the encoded omics features, first, you need to train the Vanilla AE. After you successfully trained the VanAE, you then use it for classification task, the encoded omics features will be saved in `latent_features.csv` file. (TODO: this is just a temporary solution, has to implement it as an option in arguments.)

## Sample Case: Training the Vanilla AE for cnv_methyl_rnaseq

```bash
> python run.py --autoencoder-model vanilla --omics-data ../omics_data/cnv_methyl_rnaseq.csv --train-autoencoder


===== starting =====
checkpoint file saved at ./output/checkpoints/020722-224829_vanilla/
log file save as ./output/logs/020722-224829_vanilla/
cnv_methyl_rnaseq.csv contains 292 patients with 67622 features
===== Train autoencoder =====
Epoch 1, Loss 0.02302177
Epoch 2, Loss 0.01344001
Epoch 3, Loss 0.01329384
Epoch 4, Loss 0.01316457
Epoch 5, Loss 0.01302928
Epoch 6, Loss 0.01323484
Epoch 7, Loss 0.01257442
Epoch 8, Loss 0.01246159
Epoch 9, Loss 0.01218851
Epoch 10, Loss 0.01167770
Epoch 11, Loss 0.01149966
Epoch 12, Loss 0.01138603
Epoch 13, Loss 0.01133273
Epoch 14, Loss 0.01131412
Epoch 15, Loss 0.01125329
Epoch 16, Loss 0.01121604
Epoch 17, Loss 0.01112084
Epoch 18, Loss 0.01112314
Epoch 19, Loss 0.01108457
Epoch 20, Loss 0.01105152
Epoch 21, Loss 0.01101606
Epoch 22, Loss 0.01094431
Epoch 23, Loss 0.01087702
Epoch 24, Loss 0.01081565
Epoch 25, Loss 0.01069459
Epoch 26, Loss 0.01061399
Epoch 27, Loss 0.01061716
Epoch 28, Loss 0.01059492
Epoch 29, Loss 0.01052785
Epoch 30, Loss 0.01048701
Epoch 31, Loss 0.01040611
Epoch 32, Loss 0.01033364
Epoch 33, Loss 0.01038575
Epoch 34, Loss 0.01030572
Epoch 35, Loss 0.01024863
Epoch 36, Loss 0.01023643
Epoch 37, Loss 0.01014560
Epoch 38, Loss 0.01011966
Epoch 39, Loss 0.01011356
Epoch 40, Loss 0.01000514
Epoch 41, Loss 0.00993057
Epoch 42, Loss 0.00987138
Epoch 43, Loss 0.00983969
Epoch 44, Loss 0.00977252
Epoch 45, Loss 0.00984859
Epoch 46, Loss 0.00975290
Epoch 47, Loss 0.00970984
Epoch 48, Loss 0.00970946
Epoch 49, Loss 0.00960752
Epoch 50, Loss 0.00956539
```

``` bash
> python run.py --autoencoder-model vanilla --classifier-model all --load-autoencoder ./output/checkpoints/020722-224829_vanilla/epoch_49 --biomed-data ./data/biomed_85features.csv --merged-data ./data/cnv_methyl_rnaseq_biomed_85features_121321-004411.csv --train-classifier --classifier-data embed_omics --no-save


===== starting =====
cnv_methyl_rnaseq.csv contains 292 patients with 67622 features
===== train classifier =====
===== classifier preprocess =====
cnv_methyl_rnaseq_biomed_85features_121321-004411.csv contains omics data for 234 patients with 67622 features
===== finish omics encoding =====
X_train:(187, 128)
 X_test:(47, 128)
 Y_train: (187,)
 Y_test: (47,)
===== start XGB =====
/gpfs/home/chuang25/pythonenv/tf_gpu/lib/python3.7/site-packages/xgboost/sklearn.py:1224: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
  warnings.warn(label_encoder_deprecation_msg, UserWarning)
[23:05:23] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'binary:logistic' was changed from 'error' to 'logloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
===== start RandomForest =====
===== start LogisticRegression =====
/gpfs/home/chuang25/pythonenv/tf_gpu/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
===== start SVC =====
===== start plotting results =====
/gpfs/home/chuang25/pythonenv/tf_gpu/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function `plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: RocCurveDisplay.from_predictions or RocCurveDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)
/gpfs/home/chuang25/pythonenv/tf_gpu/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function `plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: RocCurveDisplay.from_predictions or RocCurveDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)
/gpfs/home/chuang25/pythonenv/tf_gpu/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function `plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: RocCurveDisplay.from_predictions or RocCurveDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)
/gpfs/home/chuang25/pythonenv/tf_gpu/lib/python3.7/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function plot_roc_curve is deprecated; Function `plot_roc_curve` is deprecated in 1.0 and will be removed in 1.2. Use one of the class methods: RocCurveDisplay.from_predictions or RocCurveDisplay.from_estimator.
  warnings.warn(msg, category=FutureWarning)
Acc for XGBoost: 0.55
Acc for RandomForest: 0.57
Acc for LogisticRegression: 0.64
Acc for SVC: 0.68
```
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# diable all debugging logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

import numpy as np
import hyperparameters as hp
from autoencoders import (
    vanilla_autoencoder,
    variational_autoencoder,
    convolutional_autoencoder,
)
from classifiers import vanilla_classifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve

import matplotlib.pyplot as plt
import matplotlib
from utils import *


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="arguments parser for models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--autoencoder-model",
        default="vanilla",
        help="types of model to use, vanilla, convolutional, variational",
    )
    parser.add_argument(
        "--classifier-model",
        default="all",
        help="types of model to use, all, xgb, rforest, logreg",
    )
    parser.add_argument(
        "--omics-data",
        default=".." + os.sep + "omics_data" + os.sep + "cnv_methyl_rnaseq.csv",
        help="omics data file name",
    )
    parser.add_argument("--biomed-data", default=None, help="biomed data file name")
    parser.add_argument("--merged-data", default=None, help="merged data file name")
    parser.add_argument(
        "--load-autoencoder",
        default=None,
        help="path to model checkpoint file, should be similar to ./output/checkpoints/041721-201121/epoch19",
    )
    parser.add_argument(
        "--train-autoencoder", action="store_true", help="train the autoencoder"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="not save the checkpoints, logs and model. Used only for develop purpose",
    )
    parser.add_argument(
        "--train-classifier",
        default="merged",
        help="data for train the classifier, merged or biomed",
    )

    return parser.parse_args()


class CustomModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir):
        super(CustomModelSaver, self).__init__()
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, epoch, logs=None):
        save_name = "epoch_{}".format(epoch)
        tf.keras.models.save_model(
            self.model, self.checkpoint_dir + os.sep + save_name, save_format="tf"
        )


def autoencoder_loss_fn(model, input_features):
    decode_error = tf.losses.mean_squared_error(model(input_features), input_features)
    return decode_error


def autoencoder_train(loss_fn, model, optimizer, input_features, train_loss):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, input_features)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)


"""
def classifier_loss_fn(model, input_features, label_features):
    pred = model(input_features)
    pred = tf.reshape(pred, (-1,))
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(pred, label_features)
    return loss, pred


def classifier_train(
    loss_fn, model, optimizer, input_features, label_features, train_loss
):
    with tf.GradientTape() as tape:
        loss, pred = loss_fn(model, input_features, label_features)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
    return pred
"""


def main():
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    checkpoint_path = "./output/checkpoints" + os.sep + timestamp + os.sep
    logs_path = "./output/logs" + os.sep + timestamp + os.sep

    if not ARGS.no_save:
        print("checkpoint file saved at {}".format(checkpoint_path))
        print("log file save as {}".format(logs_path))

    logs_path = os.path.abspath(logs_path)

    if (
        not os.path.exists(checkpoint_path)
        and not os.path.exists(logs_path)
        and ARGS.train_autoencoder
    ):
        os.makedirs(checkpoint_path)
        os.makedirs(logs_path)

    omics_data = pd.read_csv(ARGS.omics_data, index_col=0).T.astype("float32")
    (num_patients, num_features) = omics_data.shape
    print(
        "{} contains {} patients with {} features".format(
            ARGS.omics_data.split("/")[-1], num_patients, num_features
        )
    )

    tf.convert_to_tensor(omics_data)
    omics_data = tf.expand_dims(omics_data, axis=1)
    training_dataset = tf.data.Dataset.from_tensor_slices(omics_data)
    training_dataset = training_dataset.batch(hp.batch_size)
    training_dataset = training_dataset.shuffle(num_patients)
    training_dataset = training_dataset.prefetch(hp.batch_size * 4)

    if ARGS.autoencoder_model == "vanilla":
        autoencoder = vanilla_autoencoder(
            latent_dim=hp.latent_dim,
            intermediate_dim=hp.intermediate_dim,
            original_dim=num_features,
        )
    elif ARGS.autoencoder_model == "convolutional":
        autoencoder = convolutional_autoencoder(
            latent_dim=hp.latent_dim, original_dim=num_features,
        )
    elif ARGS.autoencoder_model == "variational":
        autoencoder = variational_autoencoder(
            original_dim=num_features,
            intermediate_dim=hp.intermediate_dim,
            latent_dim=hp.latent_dim,
        )
    else:
        sys.exit("Wrong model for autoencoder!")

    if ARGS.load_autoencoder is not None:
        autoencoder.load_weights(ARGS.load_autoencoder).expect_partial()

    if ARGS.train_autoencoder:
        optimizer = tf.keras.optimizers.Adam(
            (
                tf.keras.optimizers.schedules.InverseTimeDecay(
                    hp.learning_rate, decay_steps=1, decay_rate=5e-5
                )
            )
        )

        train_loss = tf.keras.metrics.Mean("train_loss", dtype=tf.float32)

        for epoch in range(hp.num_epochs):
            for step, batch_features in enumerate(training_dataset):
                autoencoder_train(
                    autoencoder_loss_fn,
                    autoencoder,
                    optimizer,
                    batch_features,
                    train_loss,
                )
            tf.summary.scalar("loss", train_loss.result(), step=epoch)
            if not ARGS.no_save:
                save_name = "epoch_{}".format(epoch)
                autoencoder.save_weights(
                    filepath=checkpoint_path + os.sep + save_name, save_format="tf"
                )
            template = "Epoch {}, Loss {:.8f}"
            tf.print(
                template.format(epoch + 1, train_loss.result()),
                output_stream="file://{}/loss.log".format(logs_path),
            )
            print(template.format(epoch + 1, train_loss.result()))
            train_loss.reset_states()

    if ARGS.train_classifier:
        print("===== start classifier preprocess =====")
        if ARGS.train_classifier == "merged":
            merged_df = pd.read_csv(ARGS.merged_data, index_col=0)
            X, Y = merged_df.iloc[:, :-1], merged_df.iloc[:, -1]
            tf.convert_to_tensor(X)
            tf.convert_to_tensor(Y)

            # clinical data contains 41 features
            X_omics = X[:, :-41]
            X_clinical = X[:, -41:]
            X_omics = autoencoder.encoder(X_omics)
            X = tf.concat([X_omics, X_clinical], axis=2)
            X, Y = (
                X.numpy().reshape(-1, hp.intermediate_dim + 41),
                Y.numpy().reshape(-1,),
            )
        elif ARGS.train_classifier == "biomed":
            # biomed_df = pd.read_csv()
            pass

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        print(
            "X_train:{} \n X_test:{}\n Y_train: {}\n Y_test: {}".format(
                X_train.shape, X_test.shape, y_train.shape, y_test.shape
            )
        )
        if ARGS.classifier_model == "vanilla_nn":
            classifier = vanilla_classifier(input_shape=[hp.intermediate_dim])

        elif ARGS.classifier_model == "xgboost":
            xgb = XGBClassifier()

            xgb_params = {
                "booster": ["gbtree", "gblinear", "dart"],
                "n_estimators": [50, 100, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.1, 0.01, 0.001],
            }

            xgb_cv_model = GridSearchCV(
                xgb, xgb_params, cv=3, n_jobs=-1, verbose=2
            ).fit(X_train, y_train)
            print(xgb_cv_model.best_params_)
            xgb_tuned = XGBClassifier(xgb_cv_model.best_params_).fit(X_train, y_train)
            xgb_disp = plot_roc_curve(xgb, X_test, y_test)
            plt.show()

            """
            classifier.fit(X_train, y_train)
            classifier.score(X_test, y_test)
            pred_prob = classifier.predict_proba(X_test)
            pred_prob = [point[1] for point in pred_prob]

            fpr, tpr, thresholds = roc_curve(y_test, pred_prob)
            plot_roc_curve(classifier, X_test, y_test)
            plt.show()
            print("Thresholds: ")
            print(thresholds)
            """

        elif ARGS.classifier_model == "logisticregression":
            classifier = LogisticRegression()
            hyparams_logreg = dict(
                penalty=["l2", "none"],
                C=np.linespace(0.001, 1000),
                solver=["newton-cg", "lbfgs", "sag"],
                max_iter=[5000],
            )
        elif ARGS.classifier_model == "randomforest":
            classifier = RandomForestClassifier()
            hyparams_forest = dict(
                n_estimators=list(range(50, 1050, 50)),
                criterion=["gini", "entropy"],
                max_depth=list(range(1, 101)),
                min_samples_split=list(range(1, 101)),
                max_features=["sqrt", "log2", "None"],
                boostrap=[True, False],
            )
        elif ARGS.classifier_model == "all":
            print("===== start XGB =====")
            xgb = XGBClassifier(random_state=42)
            xgb.fit(X_train, y_train)
            print("===== start RandomForest =====")
            forest = RandomForestClassifier(random_state=42)
            forest.fit(X_train, y_train)
            print("===== start LogisticRegression =====")
            logreg = LogisticRegression(random_state=42)
            logreg.fit(X_train, y_train)
            print("===== start SVC =====")
            svc = SVC(random_state=42)
            svc.fit(X_train, y_train)

            print("===== start plotting results =====")

            font = {"family": "normal", "weight": "bold", "size": 10}
            matplotlib.rc("font", **font)

            xgb_disp = plot_roc_curve(xgb, X_test, y_test)
            forest_disp = plot_roc_curve(forest, X_test, y_test, ax=xgb_disp.ax_)
            logreg_disp = plot_roc_curve(logreg, X_test, y_test, ax=xgb_disp.ax_)
            svc_disp = plot_roc_curve(svc, X_test, y_test, ax=xgb_disp.ax_)
            # svc_disp.figure_.suptitle("ROC curve comparison")

            plt.savefig("omics_clinical_classifers.png", dpi=300)

        elif ARGS.classifier_model == "all_optimization":

            print("===== start xgb optimization =====")
            xgb = XGBClassifier()
            hyparams_xgb = dict(
                booster=["gbtree", "gblinear", "dart"],
                eta=np.linspace(0, 1),
                max_depth=list(range(1, 51)),
                min_child_weight=list(range(1, 51)),
                subsample=np.linspace(0, 1),
                colsample_bytree=np.linspace(0, 1),
            )
            best_xgb = rs_cv_fit_score(xgb, hyparams_xgb, X_train, y_train)
            y_pred_xgb, tpr_xgb, fpr_xgb, roc_auc_xgb = plot_roc_curves(
                best_xgb, X_test, y_test
            )
            print("===== start xgb optimization =====")
            plot_conf_int(y_pred_xgb, y_test)
            show_classification_report(best_xgb, X_test, y_test)

            print("===== start randomforest optimization =====")
            forest = RandomForestClassifier()
            hyparams_forest = dict(
                n_estimators=list(range(50, 1050, 50)),
                criterion=["gini", "entropy"],
                max_depth=list(range(1, 101)),
                min_samples_split=list(range(1, 101)),
                max_features=["sqrt", "log2", "None"],
                bootstrap=[True, False],
            )
            best_forest = rs_cv_fit_score(forest, hyparams_forest, X_train, y_train)
            y_pred_forest, tpr_forest, fpr_forest, roc_auc_forest = plot_roc_curves(
                best_forest, X_test, y_test
            )
            show_classification_report(best_forest, X_test, y_test)

            print("===== start logisticregression optimization =====")
            logreg = LogisticRegression()
            hyparams_logreg = dict(
                penalty=["l2", "none"],
                C=np.linspace(0.001, 1000),
                solver=["newton-cg", "lbfgs", "sag"],
                max_iter=[5000],
            )
            best_logreg = rs_cv_fit_score(logreg, hyparams_logreg, X_train, y_train)
            y_pred_logreg, tpr_logreg, fpr_logreg, roc_auc_logreg = plot_roc_curves(
                best_logreg, X_test, y_test
            )
            plot_conf_int(y_pred_logreg, y_test, y_test)
            show_classification_report(best_logreg, X_test, y_test)

            print("===== start plotting results =====")
            scores = pd.DataFrame()
            scores["Model"] = ["Logistic Regression", "Random Forest", "XGBoost"]
            scores["Accuracy"] = [
                round(best_logreg.best_score_, 4),
                round(best_forest.best_score_, 4),
                round(best_xgb.best_score_, 4),
            ]
            scores.sort_values(by="Accuracy", ascending=False)
            roc_no_skill = [0 for entry in range(len(y_test))]
            roc_auc_no_skill = roc_auc_score(y_test, roc_no_skill)
            fpr_no_skill, tpr_no_skill, thresholds_no_skill = roc_curve(
                y_test, roc_no_skill
            )
            plt.plot(fpr_no_skill, tpr_no_skill, linestyle="--")
            plt.plot(
                fpr_logreg,
                tpr_logreg,
                label="AUC=" + str(round(roc_auc_logreg, 4)) + " - Log Reg",
            )
            plt.plot(
                fpr_forest,
                tpr_forest,
                label="AUC=" + str(round(roc_auc_forest, 4)) + " - Rand For",
            )
            plt.plot(
                fpr_xgb,
                tpr_xgb,
                label="AUC=" + str(round(roc_auc_xgb, 4)) + " - XGBoost",
            )
            plt.title("ROC Curves")
            plt.ylabel("True Positive Rate")
            plt.xlabel("False Positive Rate")
            plt.legend(loc=4)
            plt.show()

            skf = StratifiedKFold(n_splits=4, shuffle=True)
            models = [("lr", best_logreg), ("rf", best_forest), ("xgb", best_xgb)]
            stacked_clf = StackingClassifier(
                estimators=models, final_estimator=XGBClassifier()
            )
            cv_scores = cross_val_score(
                stacked_clf, X_train, y_train, scoring="accuracy", cv=skf
            )
            cv_scores
            print("Accuracy - %0.2f" % (cv_scores.mean()))

        else:
            sys.exit("Wrong model for classifier!")

        """
        training_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        training_dataset = training_dataset.batch(hp.batch_size)
        training_dataset = training_dataset.shuffle(merged_df.shape[0])
        training_dataset = training_dataset.prefetch(hp.batch_size * 4)

        # with writer.as_default():
        #     with tf.summary.record_if(True):
        for epoch in range(hp.num_epochs):
            preds, labels = [], []
            for step, batch_features in enumerate(training_dataset):
                pred = classifier_train(
                    classifier_loss_fn,
                    classifier,
                    optimizer,
                    code,
                    batch_features[1],
                    train_loss,
                )
                preds.extend(pred.numpy().tolist())
                labels.extend(batch_features[1].numpy().tolist())
            preds = np.array(preds).reshape(-1)
            labels = np.array(labels).reshape(-1)
            acc = np.sum(preds == labels) / len(preds)
            tf.summary.scalar("loss", train_loss.result(), step=epoch)

            template = "Epoch {}, Loss: {:.8f}, Acc: {:.4f}"
            tf.print(
                template.format(epoch + 1, train_loss.result(), acc),
                output_stream="file://{}/loss.log".format(logs_path),
            )

            train_loss.reset_states()
        """


if __name__ == "__main__":
    ARGS = parse_args()
    main()

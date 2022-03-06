from email.policy import default
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# diable all debugging logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

import tensorflow as tf

import numpy as np
import hyperparameters as hp
from autoencoders import (
    vanilla_autoencoder,
    variational_autoencoder,
    convolutional_autoencoder,
)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve

import matplotlib.pyplot as plt
import matplotlib

# from utils import *


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
        help="types of model to use, all, xgb, randomforest, logreg",
    )
    parser.add_argument("--omics-data", default=None, help="omics data file name")
    parser.add_argument("--biomed-data", default=None, help="biomed data file name")
    parser.add_argument("--merged-data", default=None, help="merged data file name")
    parser.add_argument(
        "--load-autoencoder",
        default=None,
        help="path to model checkpoint file, should be similar to ./output/checkpoints/041721-201121/epoch19",
    )
    parser.add_argument(
        "--train-autoencoder", action="store_true", help="train the autoencoder model"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="not save the checkpoints, logs and model. Used only for develop purpose",
    )
    parser.add_argument(
        "--train-classifier", action="store_true", help="train the classifier model"
    )
    parser.add_argument(
        "--classifier-data",
        default=None,
        help="merged, omics, biomed, encoded_omics. The encoding process will take place during the classification process. So even when choose encoded_omics, only the raw omics data is required as input.",
    )
    parser.add_argument(
        "--save-encoded-omics",
        action="store_true",
        help="save the encoded omics data features",
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


def main():
    print("===== starting =====")
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    print("===== loading omics data =====")
    omics_data = pd.read_csv(ARGS.omics_data, index_col=0).T.astype("float32")
    (num_patients, num_features) = omics_data.shape
    print(
        "{} contains {} patients with {} features".format(
            ARGS.omics_data.split("/")[-1], num_patients, num_features
        )
    )            

    checkpoint_path = (
        "./output"
        + os.sep
        + "{}_{}_{}".format(timestamp, ARGS.autoencoder_model, ARGS.omics_data.split("/")[-1][:-4])
        + os.sep
        + "checkpoints"
    )
    logs_path = (
        "./output"
        + os.sep
        + "{}_{}_{}".format(timestamp, ARGS.autoencoder_model, ARGS.omics_data.split("/")[-1][:-4])
        + os.sep
        + "logs"
    )

    if not ARGS.no_save:
        print("checkpoint file saved at {}".format(checkpoint_path))
        print("log file save as {}".format(logs_path))

    logs_path = os.path.abspath(logs_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    if (
        not os.path.exists(checkpoint_path)
        and not os.path.exists(logs_path)
        and ARGS.train_autoencoder
    ):
        os.makedirs(checkpoint_path)
        os.makedirs(logs_path)

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
        print("===== Loading pretrained autoencoder =====")
        autoencoder.load_weights(ARGS.load_autoencoder).expect_partial()
        print("loading pretrained model at {}".format(ARGS.load_autoencoder))

    if ARGS.train_autoencoder:
        print("===== Train autoencoder =====")
        tf.convert_to_tensor(omics_data)
        omics_data = tf.expand_dims(omics_data, axis=1)
        training_dataset = tf.data.Dataset.from_tensor_slices(omics_data)
        training_dataset = training_dataset.batch(hp.batch_size)
        training_dataset = training_dataset.shuffle(num_patients)
        training_dataset = training_dataset.prefetch(hp.batch_size * 4)

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
        print("===== train classifier =====")
        print("classifier data: {}".format(ARGS.classifier_data))
        print("===== classifier preprocess =====")

        if ARGS.classifier_data == "merged":
            biomed_df = pd.read_csv(ARGS.biomed_data, index_col=0)
            num_biomed_features = biomed_df.shape[1] - 1

            merged_df = pd.read_csv(ARGS.merged_data, index_col=0).astype("float32")

            print(
                "{} contains {} patients with {} features".format(
                    ARGS.merged_data.split("/")[-1],
                    merged_df.shape[0],
                    merged_df.shape[1] - 1,
                )
            )
            X, Y = merged_df.iloc[:, :-1], merged_df.iloc[:, -1]
            tf.convert_to_tensor(X)
            tf.convert_to_tensor(Y)
            X = tf.expand_dims(X, axis=1)
            Y = tf.expand_dims(Y, axis=1)

            X_omics = X[:, :, num_biomed_features:]
            X_biomed = X[:, :, :num_biomed_features]
            if ARGS.autoencoder_model == "variational":
                X_omics = autoencoder.encoder(X_omics)[-1]
            else:
                X_omics = autoencoder.encoder(X_omics)
            
            X_encoded = X_omics.numpy().reshape(-1, hp.latent_dim)
            
            # TODO: save as .csv file with barcode as index

            if ARGS.save_encoded_omics:
                df = pd.DataFrame(X_encoded, index=merged_df.index)
                text = "latent_features_{}_{}".format(
                    ARGS.autoencoder_model, ARGS.omics_data.split("/")[-1],
                )
                df.to_csv(text)
                print("save encoded omics features in {}".format(text))

            print("===== finish omics encoding =====")
            X = tf.concat([X_omics, X_biomed], axis=2)
            X, Y = (
                X.numpy().reshape(-1, hp.latent_dim + num_biomed_features),
                Y.numpy().reshape(-1,),
            )
        elif ARGS.classifier_data == "biomed":
            biomed_df = pd.read_csv(ARGS.biomed_data, index_col=0)
            num_biomed_features = biomed_df.shape[1] - 1

            merged_df = pd.read_csv(ARGS.merged_data, index_col=0).astype("float32")

            print(
                "{} contains biomed data for {} patients with {} features".format(
                    ARGS.merged_data.split("/")[-1],
                    merged_df.shape[0],
                    num_biomed_features,
                )
            )

            X, Y = merged_df.iloc[:, :-1], merged_df.iloc[:, -1]
            tf.convert_to_tensor(X)
            tf.convert_to_tensor(Y)
            X = tf.expand_dims(X, axis=1)
            Y = tf.expand_dims(Y, axis=1)

            # X_biomed = X[:, :, -num_biomed_features:]
            X_biomed = X[:, :, :num_biomed_features]
            
            
            X, Y = (
                X_biomed.numpy().reshape(-1, num_biomed_features),
                Y.numpy().reshape(-1,),
            )

        elif ARGS.classifier_data == "omics":
            biomed_df = pd.read_csv(ARGS.biomed_data, index_col=0)
            num_biomed_features = biomed_df.shape[1] - 1

            merged_df = pd.read_csv(ARGS.merged_data, index_col=0).astype("float32")

            print(
                "{} contains omics data for {} patients with {} features".format(
                    ARGS.merged_data.split("/")[-1],
                    merged_df.shape[0],
                    merged_df.shape[1] - 1 - num_biomed_features,
                )
            )

            X, Y = (
                merged_df.iloc[:, : -1 - num_biomed_features].to_numpy(),
                merged_df.iloc[:, -1].to_numpy(),
            )

        elif ARGS.classifier_data == "encoded_omics":
            biomed_df = pd.read_csv(ARGS.biomed_data, index_col=0)
            num_biomed_features = biomed_df.shape[1] - 1

            merged_df = pd.read_csv(ARGS.merged_data, index_col=0).astype("float32")

            print(
                "{} contains omics data for {} patients with {} features".format(
                    ARGS.merged_data.split("/")[-1],
                    merged_df.shape[0],
                    merged_df.shape[1] - 1 - num_biomed_features,
                )
            )

            X, Y = merged_df.iloc[:, :-1], merged_df.iloc[:, -1]
            tf.convert_to_tensor(X)
            tf.convert_to_tensor(Y)
            X = tf.expand_dims(X, axis=1)
            Y = tf.expand_dims(Y, axis=1)

            X_omics = X[:, :, num_biomed_features:]
            if ARGS.autoencoder_model == "variational":
                X_omics = autoencoder.encoder(X_omics)[-1]
            else:
                X_omics = autoencoder.encoder(X_omics)

            X_encoded = X_omics.numpy().reshape(-1, hp.latent_dim)
            if ARGS.save_encoded_omics:
                df = pd.DataFrame(X_encoded, index=merged_df.index)
                text = "latent_features_{}_{}".format(
                    ARGS.autoencoder_model, ARGS.omics_data.split("/")[-1],
                )
                df.to_csv(text)
                print("save encoded omics features in {}".format(text))

            print("===== finish omics encoding =====")
            X, Y = (
                X_omics.numpy().reshape(-1, hp.latent_dim),
                Y.numpy().reshape(-1,),
            )
        else:
            sys.exit("wrong classifier data!")

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        print(
            "X_train:{} \n X_test:{}\n Y_train: {}\n Y_test: {}".format(
                X_train.shape, X_test.shape, y_train.shape, y_test.shape
            )
        )

        if ARGS.classifier_model == "all":
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

            font = {"weight": "bold", "size": 10}
            matplotlib.rc("font", **font)

            xgb_disp = plot_roc_curve(xgb, X_test, y_test)
            forest_disp = plot_roc_curve(forest, X_test, y_test, ax=xgb_disp.ax_)
            logreg_disp = plot_roc_curve(logreg, X_test, y_test, ax=xgb_disp.ax_)
            svc_disp = plot_roc_curve(svc, X_test, y_test, ax=xgb_disp.ax_)

            print("Acc for XGBoost: {:.2f}".format(xgb.score(X_test, y_test)))
            print("Acc for RandomForest: {:.2f}".format(forest.score(X_test, y_test)))
            print(
                "Acc for LogisticRegression: {:.2f}".format(
                    logreg.score(X_test, y_test)
                )
            )
            print("Acc for SVC: {:.2f}".format(svc.score(X_test, y_test)))

            if ARGS.classifier_data == "biomed" or ARGS.classifier_data == "omics":
                plt.savefig(
                    "./{}_{}.png".format(ARGS.autoencoder_model, ARGS.classifier_data),
                    dpi=300,
                )
            elif ARGS.classifier_data == "merged":
                plt.savefig(
                    "./{}_{}.png".format(
                        ARGS.autoencoder_model, ARGS.merged_data.split("/")[-1][:-4]
                    ),
                    dpi=300,
                )

        else:
            sys.exit("Wrong model or not implemented yet for classifier!")


if __name__ == "__main__":
    ARGS = parse_args()
    main()

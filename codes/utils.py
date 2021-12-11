from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    auc,
    precision_recall_curve,
)
from datetime import datetime


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    else:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            "\n Time take: {} hours {} minutes and {} seconds".format(
                thour, tmin, round(tsec, 2)
            )
        )


def rs_cv_fit_score(model, hyparams, X_train, y_train):
    best_clf = GridSearchCV(model, hyparams, scoring="accuracy", n_jobs=-1, cv=5)
    best_clf.fit(X_train, y_train)
    print("Best Parameters - ", best_clf.best_params_)
    print("Accuracy - %0.2f" % best_clf.best_score_ * 100)
    return best_clf


def plot_roc_curves(best_clf, X_test, y_test):
    y_predict_prob = best_clf.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_predict_prob)
    roc_auc = roc_auc_score(y_test, y_predict_prob)

    plt.plot(fpr, tpr, label="AUC=" + str(round(roc_auc, 2)))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()
    return y_predict_prob, tpr, fpr, roc_auc


# plots 95 % confidence interval for AUC ROC
def plot_conf_int(y_predict_prob, y_test):
    y_pred = np.array(y_predict_prob)
    y_true = np.array(y_test)

    print("Original ROC area: {:0.3f}".format(roc_auc_score(y_true, y_pred)))

    n_bootstraps = 1000
    rng_seed = 42
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

    plt.hist(bootstrapped_scores, bins=50)
    plt.title("Histogram of the Bootstrapped ROC AUC scores")
    plt.show()

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    print(
        "Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper
        )
    )


# plots precision recall curve
def plot_pr_curve(y_predict_prob, y_test):
    precision, recall, thresholds = precision_recall_curve(y_test, y_predict_prob)
    pr_auc = auc(recall, precision)
    pr_no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [pr_no_skill, pr_no_skill], linestyle="--")
    plt.plot(recall, precision, label="AUC=" + str(round(pr_auc, 2)))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc=4)
    plt.show()
    return precision, recall, pr_auc


# outputs classification report
def show_classification_report(best_clf):
    y_predict = best_clf.predict(X_test)
    confusion_matrix(y_test, y_predict)
    print(classification_report(y_test, y_predict))

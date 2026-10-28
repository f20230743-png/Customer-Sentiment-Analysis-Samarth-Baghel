from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

def split_data(X, y):

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


# ======================================================
# LOGISTIC REGRESSION
# ======================================================

def train_logistic_regression(X_train, y_train):

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model


# ======================================================
# NAIVE BAYES
# ======================================================

def train_naive_bayes(X_train, y_train):

    model = MultinomialNB()

    model.fit(X_train, y_train)

    return model


# ======================================================
# LINEAR SVM
# ======================================================

def train_linear_svm(X_train, y_train):

    model = LinearSVC()

    model.fit(X_train, y_train)

    return model


# ======================================================
# RANDOM FOREST
# ======================================================

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
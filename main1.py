import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Initialize global variables
metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "labels": []}
X, Y = None, None
X_train, X_test, y_train, y_test = None, None, None, None
dataset = None

st.set_page_config(layout="wide")
st.title("🧠 Software Bug Prediction with Machine Learning")

uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

if uploaded_file:
    dataset = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully.")
    st.write("### 📊 Dataset Head")
    st.dataframe(dataset.head())

    st.write("### 🔍 Missing Data Overview")
    missing_data = dataset.isnull().sum()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=missing_data.index, y=missing_data.values, ax=ax)
    ax.set_ylabel("Missing Count")
    ax.set_title("Missing Data per Feature")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig)

    st.write("### 📈 Feature Distributions")
    nunique = dataset.nunique()
    cols = [col for col in dataset.columns if 1 < nunique[col] < 50]
    rows = math.ceil(len(cols) / 5)
    fig, axs = plt.subplots(rows, 5, figsize=(25, 4 * rows))
    axs = axs.flatten()
    for i, col in enumerate(cols):
        if dataset[col].dtype == 'object':
            dataset[col].value_counts().plot.bar(ax=axs[i])
        else:
            dataset[col].hist(ax=axs[i])
        axs[i].set_title(col)
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("⚙️ Data Preprocessing")
    dataset.fillna(0, inplace=True)
    try:
        cols = ['QualifiedName', 'Name', 'Complexity', 'Coupling', 'Size', 'Lack of Cohesion']
        le = LabelEncoder()
        for col in cols:
            dataset[col] = pd.Series(le.fit_transform(dataset[col].astype(str)))
        Y = dataset.values[:, 2]
        dataset.drop(['Complexity'], axis=1, inplace=True)
        X = dataset.values
        X = normalize(X)
        if "Coupling" in dataset.columns:
            coupling_mean = dataset["Coupling"].mean()
            if coupling_mean > dataset["Coupling"].quantile(0.75):
                st.warning(f"⚠️ Coupling is high (mean = {coupling_mean:.2f}) — possible design issues!")

        if "Size" in dataset.columns:
            size_mean = dataset["Size"].mean()
            if size_mean > dataset["Size"].quantile(0.75):
                st.warning(f"⚠️ Size is high (mean = {size_mean:.2f}) — possible complexity risk!")

        if "Lack of Cohesion" in dataset.columns:
            cohesion_mean = dataset["Lack of Cohesion"].mean()
            if cohesion_mean > dataset["Lack of Cohesion"].quantile(0.75):
                st.warning(f"⚠️ Lack of Cohesion is high (mean = {cohesion_mean:.2f}) — potential design flaw!")

        st.success("✅ Preprocessing completed.")
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {e}")

    st.subheader("📉 Feature Selection with PCA")
    pca = PCA(n_components=30)
    X = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    st.text(f"Total features after PCA: {X.shape[1]}")
    st.text(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    fig, ax = plt.subplots(figsize=(18, 10))
    corr_matrix = pd.DataFrame(X_train, columns=[f'PC{i+1}' for i in range(X_train.shape[1])]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap="rocket", fmt=".2f", ax=ax, annot_kws={"size": 8})
    ax.set_title("PCA Feature Correlation Heatmap", fontsize=18)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)

    st.subheader("🚀 Train ML Models")

    def evaluate_model(name, model, Xfit, yfit):
        model.fit(Xfit, yfit)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            low_confidence = np.max(probas, axis=1) < 0.6
            low_conf_count = np.sum(low_confidence)
            if low_conf_count > 0:
                st.warning(f"⚠️ {name} predicted {low_conf_count} samples with low confidence (< 60%)")
        if 1 in pred:
            buggy_count = np.sum(pred == 1)
            st.error(f"❗ {name} detected {buggy_count} potentially buggy modules!")
            if buggy_count >= 10:
                st.error(f"🔥 Critical Alert: High number of buggy predictions ({buggy_count}) detected by {name}!")
        acc = accuracy_score(y_test, pred) * 100
        prec = precision_score(y_test, pred, average='macro', zero_division=0) * 100
        rec = recall_score(y_test, pred, average='macro', zero_division=0) * 100
        f1 = f1_score(y_test, pred, average='macro', zero_division=0) * 100
        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["labels"].append(name)
        st.text(f"{name} → Acc: {acc:.2f} | Prec: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")
        if acc < 70:
            st.warning(f"⚠️ Model '{name}' is underperforming: Accuracy = {acc:.2f}%")

    models = [
        ("Naive Bayes", BernoulliNB(binarize=0.0)),
        ("Decision Tree", DecisionTreeClassifier()),
        ("Random Forest", RandomForestClassifier()),
        ("Logistic Regression", LogisticRegression()),
        ("Bagging Classifier", BaggingClassifier(estimator=SVC(), n_estimators=1, random_state=0)),
        ("Gradient Boosting", GradientBoostingClassifier())
    ]

    if st.button("🔄 Run All ML Models"):
        metrics = {k: [] for k in metrics}
        for name, model in models:
            evaluate_model(name, model, X_train, y_train)

    st.subheader("🧠 Run CNN Model")
    if st.button("Run CNN"):
        Y_cat = to_categorical(Y)
        X_tr, X_te, y_tr, y_te = train_test_split(X, Y_cat, test_size=0.2)
        cnn_model = Sequential()
        cnn_model.add(Dense(512, input_shape=(X.shape[1],)))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(512))
        cnn_model.add(Activation('relu'))
        cnn_model.add(Dropout(0.3))
        n_classes = Y_cat.shape[1]
        cnn_model.add(Dense(n_classes))
        cnn_model.add(Activation('softmax'))
        cnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = cnn_model.fit(X_te, y_te, epochs=10, validation_data=(X_te, y_te), verbose=0)
        pred = cnn_model.predict(X_te)
        pred = np.argmax(pred, axis=1)
        testY = np.argmax(y_te, axis=1)
        buggy_count = np.sum(pred == 1)
        if buggy_count > 0:
            st.error(f"❗ CNN detected {buggy_count} potentially buggy modules!")
        if buggy_count >= 10:
            st.error(f"🔥 Critical Alert: High number of buggy predictions ({buggy_count}) detected by CNN!")
        acc = history.history['accuracy'][-1] * 100
        prec = precision_score(testY, pred, average='macro', zero_division=0) * 100
        rec = recall_score(testY, pred, average='macro', zero_division=0) * 100
        f1 = f1_score(testY, pred, average='macro', zero_division=0) * 100
        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["labels"].append("CNN")
        st.text(f"CNN → Acc: {acc:.2f} | Prec: {prec:.2f} | Recall: {rec:.2f} | F1: {f1:.2f}")
        if acc < 70:
            st.warning(f"⚠️ CNN is underperforming: Accuracy = {acc:.2f}%")

    st.subheader("📊 Comparison Graph")
    if len(metrics["labels"]) > 0:
        df_plot = pd.DataFrame({
            "Algorithm": metrics["labels"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"]
        }).set_index("Algorithm")

        st.bar_chart(df_plot)

        st.write("### Detailed Comparison")
        st.dataframe(df_plot.round(2))

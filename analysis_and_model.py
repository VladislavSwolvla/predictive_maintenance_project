import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

@st.cache_data
def load_data(file):
    try:
        if isinstance(file, str):
            if not os.path.exists(file):
                st.warning(f"Файл не найден: {file}")
                return None
            data = pd.read_csv(file)
        else:
            data = pd.read_csv(file)

        # Удаление ненужных колонок
        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

        # Преобразование категориальной переменной
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

        # Переименование колонок для XGBoost
        data.rename(columns={
            'Air temperature [K]': 'Air_temperature',
            'Process temperature [K]': 'Process_temperature',
            'Rotational speed [rpm]': 'Rotational_speed',
            'Torque [Nm]': 'Torque',
            'Tool wear [min]': 'Tool_wear'
        }, inplace=True)

        # Масштабирование числовых признаков
        scaler = StandardScaler()
        num_cols = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 'Torque', 'Tool_wear']
        data[num_cols] = scaler.fit_transform(data[num_cols])

        return data
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    st.subheader("ROC-AUC")
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    st.write(f"ROC-AUC: {roc_auc:.2f}")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend()
    st.pyplot(plt)
    plt.close()

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
    else:
        st.info("Или используется файл по умолчанию из папки data/")
        data = load_data("data/predictive_maintenance.csv")

    if data is None:
        st.stop()

    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        st.header(f"Модель: {name}")
        model.fit(X_train, y_train)
        evaluate_model(model, X_test, y_test)
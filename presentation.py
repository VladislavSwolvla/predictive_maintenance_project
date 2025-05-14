import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта")

    presentation_markdown = """
    # Прогнозирование отказов оборудования
    ---
    ## Введение
    - Цель: предсказать отказ оборудования (Target = 1)
    - Данные: [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)
    ---
    ## Этапы проекта
    1. Загрузка данных
    2. Предобработка
    3. Обучение моделей
    4. Оценка качества
    ---
    ## Используемые модели
    - Logistic Regression
    - Random Forest
    - XGBoost
    ---
    ## Метрики оценки
    - Accuracy
    - Confusion Matrix
    - ROC-AUC
    ---
    ## Приложение Streamlit
    - Загрузка данных пользователем
    - Выбор и обучение моделей
    - Визуализация результатов
    ---
    ## Заключение
    - Надёжное предсказание отказов
    - Возможности улучшения: балансировка классов, расширение признаков
    """

    rs.slides(
        presentation_markdown,
        height=600,
        theme="simple",
        config={"transition": "slide"}
    )
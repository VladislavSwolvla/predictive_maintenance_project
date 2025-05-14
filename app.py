import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

st.set_page_config(page_title="Предиктивное обслуживание", layout="wide")

PAGES = {
    "Анализ и модель": analysis_and_model_page,
    "Презентация": presentation_page,
}

st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти на:", list(PAGES.keys()))

page = PAGES[selection]
page()
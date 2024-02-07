import streamlit as st
import pandas as pd
import joblib  # Библиотека для сохранения пайплайнов/моделей
import sklearn
import catboost
sklearn.set_config(transform_output="pandas")

# Подгружаем ранее сохраненню модель
ml_pipeline_VC = joblib.load("ml_pipeline.pkl")


st.write(
    """
# Приложение, оценивающее риск заболевания сердца.

Заполните следующие параметры.
"""
)

# Ввод данных с помощью виджетов Streamlit
age = st.number_input("Введите ваш возраст", min_value=0, max_value=120, value=30)
gender = st.selectbox("Выберите ваш пол (M - male, F - female)", ("M", "F"))
chest_pain_type = st.selectbox(
    "Выберите тип боли в груди (TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic)",
    ("ATA", "TA", "NAP", "ASY"),
)
resting_bp = st.number_input(
    "Введите ваше артериальное давление в покое mm Hg",
    min_value=0,
    max_value=300,
    value=120,
)
cholesterol = st.number_input(
    "Введите ваш уровень холестерина mm/dl", min_value=0, max_value=1000, value=200
)
fasting_bs = st.number_input(
    "Уровень сахара в крови натощак (1: if FastingBS > 120 mg/dl, 0: otherwise)",
    min_value=0,
    max_value=1,
    value=0,
)
resting_ecg = st.selectbox(
    "Выберите результат ЭКГ в покое (Normal, ST, LVH)", ("Normal", "ST", "LVH")
)
max_hr = st.number_input(
    "Введите ваш максимальный пульс (Numeric value between 60 and 202)",
    min_value=60,
    max_value=202,
    value=120,
)
exercise_angina = st.selectbox(
    "Есть ли у вас стенокардия при физической нагрузке? (Y: Yes, N: No)", ("N", "Y")
)
oldpeak = st.number_input(
    "Введите значение депрессии ST-сегмента (Numeric value measured in depression)",
    min_value=-5.0,
    max_value=10.0,
    step=0.1,
    value=0.0,
)
st_slope = st.selectbox(
    "Выберите наклон ST-сегмента (Up: upsloping, Flat: flat, Down: downsloping)",
    ("Up", "Down", "Flat"),
)

# Преобразование введенных данных в список
input_data = [
    age,
    gender,
    chest_pain_type,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope,
]

# Кнопка для добавления значений в список
if st.button("Прогнозировать наличие болезни сердца"):    
    # Age	Sex	ChestPainType	RestingBP	Cholesterol	FastingBS	RestingECG	MaxHR	ExerciseAngina	Oldpeak	ST_Slope
    data = [input_data]
    columns = [
        "Age",
        "Sex",
        "ChestPainType",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG",
        "MaxHR",
        "ExerciseAngina",
        "Oldpeak",
        "ST_Slope",
    ]
    # Преобразование введенных данных в датафрейм
    input_df = pd.DataFrame(data, columns=columns)

    # Обрабатываем условия в зависимости от предсказания модели
    if ml_pipeline_VC.predict(input_df) == 0:
        st.write(
            '<span style="font-size:24px; color:green; font-weight:bold;">Поздравляем! Вероятней всего у вас нет болезни сердца! 🥳</span>',
            unsafe_allow_html=True,
        )
    else:
        st.write(
            '<span style="font-size:24px; color:red; font-weight:bold;">Скорее всего у вас проблемы с сердцем. 😬 Обратитесь к специалисту! 🤒</span>',
            unsafe_allow_html=True,
        )
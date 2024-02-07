import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler

class LogReg:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate      
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.coef_ = np.random.normal(size = X.shape[1])
        self.intercept_ = np.random.normal()
        n_epochs = 1000
        for epoch in range(n_epochs):
            p = 1 / (1 + np.exp(-(X @ self.coef_ + self.intercept_)))
            error = (p - y)
            w0_grad = error 
            w_grad = X * error.reshape(-1, 1) # Когда мы говорим reshape(-1, 1), 
            # мы фактически преобразуем одномерный массив (вектор) в двумерный массив (матрицу) с одним столбцом. 

            self.coef_ = self.coef_ - self.learning_rate * w_grad.mean(axis=0)
            self.intercept_ = self.intercept_ - self.learning_rate * w0_grad.mean(axis=0)
    
    def predict(self, X):
        X = np.array(X)  
        p = np.round(1 / (1 + np.exp(-(X @ self.coef_ + self.intercept_))))
        return p
    # def score(self, X, y):
    #   return mean_absolute_error(y, X @ self.coef_ + self.intercept_)

st.write("## Logistic regression app")

uploaded_file = st.sidebar.file_uploader("Выберите файл", type=["csv"])

if uploaded_file is not None:
    st.sidebar.write("Файл успешно загружен!")

    df = pd.read_csv(uploaded_file)
    st.write("Данные из файла:")
    st.write(df)
    target_column = st.selectbox("Выберите целевой столбец для регрессии", df.columns)
    
    scaler = StandardScaler().set_output(transform="pandas")
    scaled_df = scaler.fit_transform(df.loc[:, df.columns != target_column])
    
    model = LogReg(0.1)
    model.fit(scaled_df, df[target_column])

    st.write("Результаты лог. регрессии:")
    if model is not None:
        weights = dict(zip(df.columns, model.coef_))
        st.write(weights)
    
    feature_1 = st.selectbox("Выберите ось x для scatter plot", df.columns)
    feature_2 = st.selectbox("Выберите ось y для scatter plot", df.columns)
    
    if model is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=feature_1, y=feature_2, hue=target_column, data=df, ax=ax)
        
        # plt.scatter(df[f'{feature_1}'], df[f'{feature_2}'], c=df[f'{target_column}'], cmap='summer')
        # cbar = plt.colorbar()
        # cbar.set_label('Color Feature')
        # ax.set_title("Scatter Plot")

        x_values = df[f'{feature_1}']
        y_values = (-model.coef_[0] * x_values - model.intercept_) / model.coef_[1]

        ax.plot(x_values, y_values, color='red', label='Линия регрессии')

        # sns.lineplot(x=x_values, y=y_values, color='red', label='Линия регрессии')
        
        st.pyplot(fig)








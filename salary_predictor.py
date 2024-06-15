import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Function to load data from CSV
def load_data(filename):
    return pd.read_csv(filename)

# Function to preprocess the data
def preprocess_data(data):
    X = data.drop(columns=['salary'])
    y = data['salary']

    numeric_features = ['years_of_experience']
    categorical_features = ['degree']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor

# Function to train a model
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Main function to display the interface and make predictions
def main():
    # Title and subheader
    st.title('Software Developer Salary Prediction')
    st.subheader('Using Machine Learning')

    # Load data
    filename = 'salaries.csv'
    data = load_data(filename)

    # Preprocess data
    X, y, preprocessor = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # User input for prediction
    st.subheader('Predict Salary')
    experience = st.slider('Years of Experience', 1, 10, 5)
    degree = st.selectbox('Degree', ['Bachelors', 'Masters', 'Phd'])

    # Convert user input to DataFrame
    user_data = pd.DataFrame([[experience, degree]], columns=['years_of_experience', 'degree'])

    # Preprocess user input
    user_data_transformed = preprocessor.transform(user_data)

    # Predict and display the salary
    prediction = model.predict(user_data_transformed)[0]
    st.write(f'Predicted Salary: ${prediction:,.2f}')

if __name__ == '__main__':
    main()

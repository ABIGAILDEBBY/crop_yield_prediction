import warnings

import joblib  # Using joblib for pickling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

plt.style.use("ggplot")

warnings.filterwarnings(
    "ignore"
)  # Suppress all warnings (Not recommended for production)


def load_data(data_path):
    """
    Loads the crop yield data from a CSV file.

    Args:
        data_path (str): Path to the CSV file containing crop yield data.

    Returns:
        pandas.DataFrame: The loaded DataFrame containing crop yield data.
    """

    df = pd.read_csv(data_path)
    df.drop(
        "Unnamed: 0", axis=1, inplace=True
    )  # Assuming 'Unnamed: 0' is an extra column
    return df


def preprocess_data(df):
    """
    Preprocesses the data by handling missing values, encoding categorical
    features,
    and scaling numerical features.

    Args:
        df (pandas.DataFrame): The DataFrame containing crop yield data.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
        sklearn.compose.ColumnTransformer: The column transformer used for
        preprocessing.
    """

    # Handle missing values (if any)
    # ... You can add code to handle missing values (e.g., imputation)

    # Select features for modeling
    col = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "Area",
        "Item",
    ]
    df = df[col]

    # Separate features (without splitting)
    print("Columns ", df.columns)
    X = df.drop("hg/ha_yield", axis=1)
    y = df["hg/ha_yield"]  # Target variable remains for reference

    # Create transformers for categorical and numerical features
    ohe = OneHotEncoder(drop="first")
    scale = StandardScaler()

    preprocesser = ColumnTransformer(
        transformers=[
            ("StandardScale", scale, [0, 1, 2, 3]),
            ("OneHotEncode", ohe, [4, 5]),
        ],
        remainder="passthrough",
    )

    # Fit the transformers on ALL data (assuming no missing values for
    # transformation)
    X = preprocesser.fit_transform(X)

    return df, X, y, preprocesser


def train_evaluate_models(X, y):
    """
    Trains and evaluates different machine learning models for yield
    prediction.

    Args:
        X_train (numpy.ndarray): The preprocessed training features.
        y_train (numpy.ndarray): The training target variable (yield).

    Returns:
        dict: A dictionary containing the trained models.
    """

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LINEAR REGRESSION": LinearRegression(),
        "LASSO": Lasso(),
        "RIDGE": Ridge(),
        "DECISION TREE": DecisionTreeRegressor(),
        "KNN": KNeighborsRegressor(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Evaluate model performance on the held-out test set
        print(
            f"{name}\n MAE: {mean_absolute_error(y_test, model.predict(X_test))} score: {r2_score(y_test, model.predict(X_test))}"
        )

    return models


def predict(model, preprocesser, features):
    """
    Predicts yield for a new data point using a trained model.

    Args:
        model (object): The trained machine learning model.
        preprocesser (sklearn.compose.ColumnTransformer): The column
        transformer used for preprocessing.
        features (list): A list of features for the new data point.

    Returns:
        float: The predicted yield in hg/ha.
    """

    features = np.array([features], dtype=object)
    transformed_features = preprocesser.transform(features)
    predicted_yield = model.predict(transformed_features).reshape(-1, 1)
    return predicted_yield[0][0]


def save_model(model, filename):
    """
    Saves the trained model using joblib for persistence.

    Args:
        model (object): The trained machine learning model.
        filename (str): The filename to save the model as.
    """

    joblib.dump(model, filename)  # Using joblib for pickling


def main():
    """
    The main function that loads data, preprocesses, trains models,
    makes predictions, and saves the model.
    """

    data_path = "yield_df.csv"  # Replace with your actual data path
    df = load_data(data_path)
    print(df.head())

    # Exploratory Data Analysis (EDA)
    numerical_features = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "hg/ha_yield",
    ]

    # Distribution plots for numerical features
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, feature in enumerate(numerical_features[: len(axes.flat)]):
        ax = axes.flat[i]
        sns.distplot(df[feature], ax=ax)
        ax.set_title(feature)
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")
    plt.tight_layout()
    plt.show()

    # Boxplots for features by Area and Item
    categorical_features = ["Area", "Item"]
    for feature in numerical_features:
        for cat in categorical_features:
            fig, ax = plt.subplots()
            sns.boxplot(x=cat, y=feature, showmeans=True, data=df)
            ax.set_title(f"{feature} by {cat}")
            plt.show()

    # Correlation matrix and heatmap
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr(method="spearman")

    plt.figure(figsize=(10, 6))

    # ... visualize the correlation matrix using heatmaps
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix of Crop Yield Features")
    plt.show()

    # Feature Engineering (Optional)
    # ... Based on EDA results, you can add code for feature engineering here

    df, X_train_dummy, y_train, preprocesser = preprocess_data(
        df.copy()
    )  # Avoid modifying original df

    models = train_evaluate_models(
        X_train_dummy, y_train
    )  # Assuming y_train is defined elsewhere (train/test split)

    # Select the best model based on your evaluation criteria
    best_model = models["DECISION TREE"]  # Replace with your chosen model

    # Make prediction on a new data point (example)
    year = 2025
    rain = 1500.0
    pesticides = 130.0
    temp = 17.5
    area = "Asia"
    item = "Wheat"

    predicted_yield = predict(
        best_model, preprocesser, [year, rain, pesticides, temp, area, item]
    )
    print(f"Predicted yield for {year} in {area} ({item}): {predicted_yield:.2f} hg/ha")

    # Save the best model for future use
    save_model(best_model, "yield_model.pkl")


if __name__ == "__main__":
    main()

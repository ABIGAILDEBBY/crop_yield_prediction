import warnings

import joblib  # Using joblib for pickling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

plt.style.use("ggplot")

warnings.filterwarnings(
    "ignore"
)  # Suppress all warnings (Not recommended for production)

models = {
    "LINEAR REGRESSION": LinearRegression(),
    "LASSO": Lasso(),
    "RIDGE": Ridge(),
    "DECISION TREE": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
}


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
    features, and scaling numerical features.

    Args:
        df (pandas.DataFrame): The DataFrame containing crop yield data.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
        Features dataset set(X): Feature columns extracted from dataframe df
        Targeted Variable (y): Targeted column from df
        sklearn.compose.ColumnTransformer: The column transformer used for
        preprocessing.
    """

    # Handle missing values (if any)
    # ... You can add code to handle missing values (e.g., imputation)
    # This dataset has no missing values for all the columns
    # Select features for modeling
    col = [
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp",
        "Item",
        "Year",
        "Area",
    ]

    y = df["hg/ha_yield"]  # Target variable
    df = df[col]
    X = df

    # Create transformers for categorical and numerical features
    ohe = OneHotEncoder(handle_unknown="ignore")
    scale = StandardScaler()

    preprocesser = ColumnTransformer(
        transformers=[
            ("StandardScale", scale, [0, 1, 2]),  # Standardize
            ("OneHotEncode", ohe, [3, 4, 5]),
        ],
        remainder="passthrough",
    )

    # Fit the transformers on ALL data (assuming no missing values for
    # transformation)
    X = preprocesser.fit_transform(X)

    return df, X, y, preprocesser


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Args:
        X (numpy.ndarray): The preprocessed features.
        y (numpy.ndarray): The target variable (yield).
        test_size (float, optional): The proportion of data for the testing set. Defaults to 0.2.
        random_state (int, optional): The random seed for splitting. Defaults to 42.

    Returns:
        tuple: A tuple containing four elements:
            1. numpy.ndarray: The training features.
            2. numpy.ndarray: The testing features.
            3. numpy.ndarray: The training target variable.
            4. numpy.ndarray: The testing target variable.
    """

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_evaluate_models(X, y):
    """
    Trains and evaluates different machine learning models for yield
    prediction using K-Fold cross-validation.

    Args:
        X (numpy.ndarray): The preprocessed features.
        y (numpy.ndarray): The target variable (yield).

    Returns:
        tuple: A tuple containing two elements:
            1. dict: A dictionary containing the trained models and their average performance metrics.
            2. object: The best performing model based on the chosen metric.
    """

    # Define number of folds for K-Fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "LINEAR REGRESSION": LinearRegression(),
        "LASSO": Lasso(),
        "RIDGE": Ridge(),
        "DECISION TREE": DecisionTreeRegressor(),
        "KNN": KNeighborsRegressor(),
    }

    model_results = {}
    best_model_name = None
    best_model_value = float("inf")  # Initialize with a high value for MAE

    for name, model in models.items():
        # Initialize lists to store evaluation metrics during cross-validation
        mae_scores = []
        r2_scores = []

        for train_index, test_index in kfold.split(X):
            # Split data into training and testing sets for each fold
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train the model on the training fold
            model.fit(X_train, y_train)

            # Evaluate model performance on the testing fold
            y_pred = model.predict(X_test)
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))

        # Calculate and store average performance metrics across all folds
        model_results[name] = {
            "MAE": np.mean(mae_scores),
            "R2": np.mean(r2_scores),
        }

        # Track the best model based on chosen metric (lower MAE or higher R2)
        chosen_metric = "R2"  # Replace with 'R2' if you prefer R-squared
        current_value = model_results[name][chosen_metric]
        if current_value < best_model_value:  # For lower MAE
            # OR if current_value > best_model_value:  # For higher R2
            best_model_name = name
            best_model_value = current_value

        # Print average performance for informational purposes (optional)
        print(
            f"{name} - Average MAE: {model_results[name]['MAE']:.4f}, Average R2: {model_results[name]['R2']:.4f}"
        )

    return models[best_model_name]


def predict(best_model, preprocesser, features):
    """
    Predicts yield for a new data point using the best model based on results.

    Args:
        models (dict): Dictionary containing the trained models.
        model_results (dict): Dictionary containing performance metrics (MAE, R2) for each model.
        preprocesser (sklearn.compose.ColumnTransformer): The column
        transformer used for preprocessing.
        features (list): A list of features for the new data point.

    Returns:
        float: The predicted yield in hg/ha.
    """

    features = np.array([features], dtype=object)
    transformed_features = preprocesser.transform(features)
    predicted_yield = best_model.predict(transformed_features).reshape(-1, 1)
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
    The main function loads data, preprocesses, trains models,
    makes predictions, and saves the model.
    """

    data_path = "csv_files/yield_df.csv"  # Replace with your actual data path
    df = load_data(data_path)

    # Exploratory Data Analysis (EDA)
    numerical_features = [
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

        # Rotate x-axis labels for better readability
        # Update x-tick positions in case they changed
        ax.set_xticks(ax.get_xticks())
        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Boxplots for features by Area and Item
    categorical_features = ["Area", "Item", "Year"]
    for feature in numerical_features:
        for cat in categorical_features:
            fig, ax = plt.subplots(figsize=(15, 7))
            sns.boxplot(x=cat, y=feature, showmeans=True, data=df)
            ax.set_title(f"{feature} by {cat}")
            # Rotate x-axis labels for better readability
            ax.set_xticks(ax.get_xticks())
            # Rotate x-axis labels
            ax.tick_params(axis="x", which="both", rotation=45)
            plt.show()

    # Correlation matrix and heatmap
    # Select only numeric columns
    numeric_cols = df[numerical_features]
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

    best_model = train_evaluate_models(
        X_train_dummy, y_train
    )  # Assuming y_train is defined elsewhere (train/test split)

    # Make prediction on a new data point (example)
    year = 2025
    rain = 1500.0
    pesticides = 130.0
    temp = 17.5
    area = "Asia"
    item = "Wheat"

    # models, model_results, preprocesser, features
    predicted_yield = predict(
        best_model, preprocesser, [rain, pesticides, temp, item, year, area]
    )
    print(f"Predicted yield for {year} in {area} ({item}): {predicted_yield:.2f} hg/ha")

    # Save the best model for future use
    save_model(best_model, "yield_model.pkl")


if __name__ == "__main__":
    main()

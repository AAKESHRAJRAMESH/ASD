# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

print("Libraries imported successfully.")

def load_and_clean_data(file_path):
    """
    Reads an .arff file, converts it to a pandas DataFrame, and decodes
    byte strings into standard Python strings for easier manipulation.
    """
    try:
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Clean byte strings by decoding them to utf-8
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode('utf-8')
        
        print(f"Dataset '{file_path}' loaded and cleaned successfully.")
        print(f"Shape of the dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

# --- 2. Preprocess Data ---
def preprocess_data(df):
    """
    Handles missing values, removes specified columns, encodes categorical 
    features, and encodes the target variable.
    """
    if df is None:
        return None, None, None, None, None

    # Replace '?' with NaN (standard missing value representation)
    df.replace('?', np.nan, inplace=True)
    
    # Separate features (X) and target (y)
    X = df.drop('Class/ASD', axis=1)
    y = df['Class/ASD']

    # Define and drop the specified columns
    columns_to_drop = ['ethnicity', 'contry_of_res', 'used_app_before', 'age_desc', 'relation']
    X = X.drop(columns=columns_to_drop)
    print(f"Removed columns: {', '.join(columns_to_drop)}")
    
    # Handle missing values
    # For numerical columns, fill with the median
    for col in X.select_dtypes(include=np.number).columns:
        X[col].fillna(X[col].median(), inplace=True)
    # For categorical columns, fill with the mode (most frequent value)
    for col in X.select_dtypes(include='object').columns:
        X[col].fillna(X[col].mode()[0], inplace=True)
        
    print("Missing values handled.")
    
    # Store original columns for interactive prediction
    original_cols = X.columns.tolist()

    # Encode categorical features using one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    encoded_cols = X_encoded.columns.tolist()
    
    # Encode the target variable ('NO' -> 0, 'YES' -> 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Categorical features and target variable encoded.")
    return X_encoded, y_encoded, original_cols, encoded_cols, df

# --- 3. Train the Model ---
def train_lgbm_model(X, y):
    """
    Splits the data into training and testing sets and trains a LightGBM classifier.
    """
    if X is None or y is None:
        return None, None, None
        
    # Split data into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize LightGBM Classifier
    model = lgb.LGBMClassifier(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("LightGBM model trained successfully.")
    return model, X_test, y_test

# --- 4. Evaluate the Model ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test set and prints performance metrics.
    """
    if model is None:
        return
        
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print the results
    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['NO (Non-ASD)', 'YES (ASD)']))
    print("------------------------\n")

# --- 5. Interactive Prediction ---
def interactive_prediction(model, original_cols, encoded_cols, original_df):
    """
    Allows a user to input data for each feature and get a real-time prediction.
    """
    if model is None:
        return

    print("--- Interactive ASD Screening Tool ---")
    print("Please answer the following questions to get a risk assessment.")
    
    user_input = {}
    
    # Collect data for the remaining features
    for col in original_cols:
        options = original_df[col].dropna().unique()

        if pd.api.types.is_numeric_dtype(original_df[col]):
             # Handle AQ-10 score questions specifically
            if 'Score' in col:
                 while True:
                    val = input(f"Enter {col} (1 for 'yes', 0 for 'no'): ")
                    if val in ['0', '1']:
                        user_input[col] = int(val)
                        break
                    else:
                        print("Invalid input. Please enter 0 or 1.")
            else: # For other numeric columns like 'age' or 'result'
                while True:
                    val = input(f"Enter value for '{col}' (e.g., {int(options.mean())}): ")
                    try:
                        user_input[col] = float(val)
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        else:
            # For categorical features, show options
            options_str = ", ".join(map(str, options))
            while True:
                val = input(f"Enter value for '{col}' (Options: {options_str}): ")
                if val in options:
                    user_input[col] = val
                    break
                else:
                    print(f"Invalid input. Please choose from the available options.")

    # --- Preprocess user input ---
    # 1. Create a DataFrame from the user's input
    input_df = pd.DataFrame([user_input])
    
    # 2. One-hot encode the input DataFrame
    input_encoded = pd.get_dummies(input_df, drop_first=True)
    
    # 3. Align columns with the training data
    input_aligned = input_encoded.reindex(columns=encoded_cols, fill_value=0)

    # --- Make Prediction ---
    prediction = model.predict(input_aligned)[0]
    probability = model.predict_proba(input_aligned)[0]
    
    print("\n--- Screening Result ---")
    if prediction == 1:
        print("Prediction: High Risk for ASD (1)")
        print("Recommendation: Refer for Stage 2 (Gaze Assessment).")
    else:
        print("Prediction: Low Risk for ASD (0)")
        print("Recommendation: No immediate further assessment required.")
    print("------------------------")


if __name__ == "__main__":
    file_path = 'Autism-Child-Data.arff'
    
    # Execute the workflow
    df = load_and_clean_data(file_path)
    X_processed, y_processed, original_cols, encoded_cols, df_cleaned = preprocess_data(df)
    model, X_test, y_test = train_lgbm_model(X_processed, y_processed)
    
    # Evaluate the final model
    evaluate_model(model, X_test, y_test)
    
    # Start the interactive part if the model trained successfully
    if model:
        interactive_prediction(model, original_cols, encoded_cols, df_cleaned)
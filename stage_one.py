import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ============================
# Step 1: Load Dataset
# ============================
file_path = "Autism-Child-Data.arff"  # change path if needed
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# ============================
# Step 2: Preprocessing Function
# ============================
def preprocess_dataframe(df, training=True):
    # Decode byte strings
    df = df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # Fill missing age with median
    if "age" in df.columns:
        df["age"] = df["age"].astype(float)
        df["age"] = df["age"].fillna(df["age"].median())

    # Encode target column
    if training and "Class/ASD" in df.columns:
        df["Class/ASD"] = df["Class/ASD"].map({"YES": 1, "NO": 0})

    return df

df_clean = preprocess_dataframe(df, training=True)

# ============================
# Step 3: Split Features/Target
# ============================
X = df_clean.drop(columns=["Class/ASD"])
y = df_clean["Class/ASD"]

# Identify categorical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# Step 4: Train LightGBM Model
# ============================
model = LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    is_unbalance=True,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    categorical_feature=categorical_cols,
    callbacks=[early_stopping(20), log_evaluation(0)]
)

# ============================
# Step 5: Evaluate
# ============================
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("Model Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["Non-ASD", "ASD"]))

# ============================
# Step 6: Predict New Input
# ============================
def predict_new_data(input_dict):
    # Convert dict to DataFrame
    new_df = pd.DataFrame([input_dict])

    # Preprocess (same as training, but no target encoding)
    new_df = preprocess_dataframe(new_df, training=False)

    # Predict
    pred = model.predict(new_df)[0]
    prob = model.predict_proba(new_df)[0][1]

    if pred == 1:
        return f"Prediction: High Risk of ASD (1), Probability={prob:.2f}"
    else:
        return f"Prediction: Low Risk / Non-ASD (0), Probability={prob:.2f}"

# ============================
# Example usage of prediction
# ============================
sample_input = {
    "A1_Score": "1", "A2_Score": "1", "A3_Score": "0", "A4_Score": "0", "A5_Score": "1",
    "A6_Score": "1", "A7_Score": "0", "A8_Score": "1", "A9_Score": "0", "A10_Score": "0",
    "age": 10, "gender": "m", "ethnicity": "Others", "jundice": "no", "austim": "no",
    "contry_of_res": "India", "used_app_before": "no", "result": 6,
    "age_desc": "4-11 years", "relation": "Parent"
}

print("\nSample Prediction ->", predict_new_data(sample_input))

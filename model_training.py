# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import pickle

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('customer_churn_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: customer_churn_data.csv not found.")
    exit()

# --- 2. Data Cleaning and Preprocessing ---
# Drop the CustomerID column
if 'CustomerID' in df.columns:
    df.drop('CustomerID', axis=1, inplace=True)

# No missing values in 'TotalCharges' based on our inspection, but keep this just in case
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print("Data cleaning complete.")

# --- 3. Feature Engineering ---
# Define the target variable
target = 'Churn'
y = df[target]

# Define features
X = df.drop(target, axis=1)

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

print("Feature encoding setup complete.")

# --- 4. Model Training ---
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 5. Model Evaluation ---
y_pred = pipeline.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 6. Save the Model and Column Order ---
model_data = {
    'pipeline': pipeline,
    'columns': X.columns.tolist()
}

with open('churn_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\nModel saved to 'churn_model.pkl'. Ready to use in the Streamlit app.")

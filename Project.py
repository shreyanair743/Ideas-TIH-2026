#--------------QUESTION 1-----------------

import pandas as pd
import zipfile
import os

#Path to the downloaded ZIP file
zip_path = r"C:\Users\shrey\Downloads\archive.zip"

# Extract the ZIP contents
extract_folder = r"C:\Users\shrey\Downloads\student_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Show files extracted from the ZIP
print("Files in ZIP:")
print(os.listdir(extract_folder))

# Replace the filename below with the actual CSV file name if needed
csv_file = os.path.join(extract_folder, "student_performance_dataset.csv")

# Load the dataset
df = pd.read_csv(csv_file)

# Display first 5 rows
df.head(5)

#--------------QUESTION 2------------------------

import pandas as pd
import zipfile
import os

# Path to the downloaded ZIP file
zip_path = r"C:\Users\shrey\Downloads\archive.zip"

# Extract the ZIP contents
extract_folder = r"C:\Users\shrey\Downloads\student_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Show files extracted from the ZIP
print("Files in ZIP:")
print(os.listdir(extract_folder))

# Replace the filename below with the actual CSV file name if needed
csv_file = os.path.join(extract_folder, "student_performance_dataset.csv")

# Load the dataset
df = pd.read_csv(csv_file)

# Numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

#--------------QUESTION 3---------------------------

print(df.isnull().sum())

#--------------QUESTION 4---------------------------

# Fill missing numerical values with median
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with mode
cat_cols = df.select_dtypes(exclude='number').columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Print total missing values remaining
print("Total missing values remaining:", df.isnull().sum().sum())

#--------------QUESTION 5-------------------------

from sklearn.preprocessing import LabelEncoder

categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

print(df.head())

#--------------QUESTION 6-------------------------

# Separate features and target
X = df.drop("performance_category", axis=1)
y = df["performance_category"]

# Print shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

#--------------QUESTION 7---------------------------

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop("performance_category", axis=1)
y = df["performance_category"]

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

#--------------QUESTION 8---------------------------

from sklearn.preprocessing import MinMaxScaler

# Split features and target
X = df.drop("performance_category", axis=1)
y = df["performance_category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize scaler
scaler = MinMaxScaler()

# Fit on training data and transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame (optional but useful)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

# Print min and max values
print("Min value in scaled X_train:", X_train_scaled.min().min())
print("Max value in scaled X_train:", X_train_scaled.max().max())

#--------------QUESTION 9------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Split features and target
X = df.drop("performance_category", axis=1)
y = df["performance_category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (optional but consistent with previous step)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

#--------------QUESTION 10----------------------------

# Split features and target
X = df.drop("performance_category", axis=1)
y = df["performance_category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_

# Convert to Series for better readability
feature_importance = pd.Series(importances, index=X.columns)

print(feature_importance)

#----------------------------------------------

#--------------QUESTION 1-----------------

import pandas as pd
df = pd.read_excel('student_performance_dataset.csv.xlsx')
print(df.head())

#--------------QUESTION 2------------------------

# Numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# Categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print("Numerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)

#--------------QUESTION 3---------------------------

missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)

#--------------QUESTION 4---------------------------

# Fill numerical columns with median
for col in df.select_dtypes(include=['number']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Check remaining missing values
remaining_missing = df.isnull().sum().sum()

print("Total missing values remaining:", remaining_missing)

#--------------QUESTION 5-------------------------

from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder object
le = LabelEncoder()

# Get categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Display first 5 rows
print(df.head())

#--------------QUESTION 6-------------------------

# Separate features and target
X = df.drop('performance_category', axis=1)
y = df['performance_category']

# Print shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

#--------------QUESTION 7---------------------------

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# Print shapes
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)

#--------------QUESTION 8---------------------------

from sklearn.preprocessing import MinMaxScaler

# Create scaler
scaler = MinMaxScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print min and max values of scaled training data
print("Minimum value in X_train_scaled:", X_train_scaled.min())
print("Maximum value in X_train_scaled:", X_train_scaled.max())

#--------------QUESTION 9------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize the model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = dt_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy Score:", accuracy)

#--------------QUESTION 10----------------------------

# Get feature importances
feature_importance = pd.Series(
    dt_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

# Print feature importances
print(feature_importance)

#----------------------------------------------

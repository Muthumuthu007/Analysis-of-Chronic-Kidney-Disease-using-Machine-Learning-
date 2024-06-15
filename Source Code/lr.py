import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformerx
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv('kidney.csv')

# Check the column names
print("Column Names:", df.columns)

# Data Preprocessing
# Assuming 'classification' is your binary outcome variable (1 for CKD, 0 for non-CKD)
# Ensure 'classification' is in the list of column names
if 'classification' in df.columns:
    X = df.drop('classification', axis=1)
    y = (df['classification'] == 'ckd').astype(int)  # Convert 'ckd' to 1, 'notckd' to 0
else:
    raise ValueError("Column 'classification' not found in the dataset.")

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.columns.difference(categorical_cols)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing to training and testing sets
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_preprocessed = imputer.fit_transform(X_train_preprocessed)
X_test_preprocessed = imputer.transform(X_test_preprocessed)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train_preprocessed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_preprocessed)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# If you want to see the coefficients and intercept
print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)

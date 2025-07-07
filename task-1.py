##### TASK-1 DATA PIPELINE FOR ETL PROCESS
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#Load Dataset
df = pd.read_csv('sample_data.csv') 

# Identify columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Create Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# ETL Pipeline
etl_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Split and Transform
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
X_train_processed = etl_pipeline.fit_transform(X_train)
X_test_processed = etl_pipeline.transform(X_test)

print(" Data preprocessing pipeline completed successfully.")
print('-'*50)
### numpy arrays
# print(X_train_processed)
# print("-" *50)
# print(X_test_processed)

# to convert it back to a DataFrame (optional)
from sklearn.compose import make_column_selector as selector

# Note: This will only work if OneHotEncoder returns dense array (not sparse)
print(pd.DataFrame(X_train_processed))
print('-'*50)
print(pd.DataFrame(X_test_processed))

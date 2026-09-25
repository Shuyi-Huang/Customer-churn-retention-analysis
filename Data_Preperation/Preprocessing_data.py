import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# 1. Load Dataset
Data = "Dataset_ATS_v2.csv"

df = pd.read_csv(Data)

# print(df['InternetService'].value_counts())
# print(df['Contract'].value_counts())

# print("Dataset Loaded Successfully")
# print("Dataset Shape:", df.shape)

# 2. Check Missing Values and Duplicates

missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)


duplicates = df.duplicated().sum()

print("\nDuplicate Records:", duplicates)

#3.convert Target variable

df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

#4. Encode Binary Categorical Columns

encoder = LabelEncoder()

for col in ['gender','Dependents','PhoneService','MultipleLines']:
    df[col] = encoder.fit_transform(df[col])
    
#One-Hot Encode Multi-Class Columns
df = pd.get_dummies(
    df,
    columns=['InternetService','Contract'],
    drop_first=False
)

#Scale Numerical Features

scaler = StandardScaler()

df[['tenure','MonthlyCharges']] = scaler.fit_transform(
    df[['tenure','MonthlyCharges']]
)

df[['tenure', 'MonthlyCharges']] = df[['tenure', 'MonthlyCharges']].apply(
    pd.to_numeric
)

print(df[['tenure', 'MonthlyCharges']].dtypes)

# Convert only boolean columns to integers
bool_cols = df.select_dtypes(include='bool').columns

df[bool_cols] = df[bool_cols].astype(int)

# # Save preprocessed dataset
# df.to_csv("customer_churn_preprocessed_Data.csv", index=False)

print(df.head())







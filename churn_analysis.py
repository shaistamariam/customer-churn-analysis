import pandas as pd 

train = pd.read_csv("Customer churn dataset training.csv")

test =pd.read_csv("Customer churn dataset testing.csv")

print("traning data")
print(train.head())

print("/ntesting data")
print(test.head())

#Check column size
print("Training data shape:", train.shape)
print("Testing data shape:", test.shape)

#Check column names
print("/n Training columns:")
print(train.columns.tolist())

print("/n Testing columns:")
print(test.columns.tolist())

#Check data type
print("\nTraining data types:")
print(train.dtypes)

print("\nTesting data types:")
print(test.dtypes)

#Check Missing values
print("\nMissing values in training data:")
print(train.isnull().sum())

print("\nMissing values in testing data:")
print(test.isnull().sum())

#Check Missing values in rows
print("\nRows with missing values in training data:")
print(train[train.isnull().any(axis=1)])

#Remove completely empty rows 
train = train.dropna(how="all")

print("\nMissing values after removing empty rows:")
print(train.isnull().sum())

print("\nCompletely empty rows in testing data:")
print(test[test.isnull().all(axis=1)])

#Check the duplicate rows 
print("\nDuplicate rows in training data:")
print(train.duplicated().sum())

print("\nDuplicate rows in testing data:")
print(test.duplicated().sum())

#Check the duplicate ID
print("\nDuplicate Customer IDs in training:")
print(train["CustomerID"].duplicated().sum())

print("\nDuplicate Customer IDs in testing:")
print(test["CustomerID"].duplicated().sum())

#Check numerical column
print("\nNumerical columns:")
print(train.describe())

#Check the Categorical column 
print("\nCategorical columns and their values:")

print("\nGender:")
print(train["Gender"].value_counts())

print("\nSubscription Type:")
print(train["Subscription Type"].value_counts())

print("\nContract Length:")
print(train["Contract Length"].value_counts())

print("\nChurn:")
print(train["Churn"].value_counts()) 

# Fix numerical data types

integer_columns = [
    "CustomerID",
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Last Interaction",
    "Churn"
]

for column in integer_columns:
    train[column] = train[column].astype(int)
    test[column] = test[column].astype(int)

# Total Spend should remain decimal
train["Total Spend"] = train["Total Spend"].astype(float)
test["Total Spend"] = test["Total Spend"].astype(float)

print("\nFinal data types after conversion:")
print(train.dtypes)

#Check unusual values
print("\nChecking unusual values:")

print("\nAge:")
print(train["Age"].describe())

print("\nTenure:")
print(train["Tenure"].describe())

print("\nUsage Frequency:")
print(train["Usage Frequency"].describe())

print("\nSupport Calls:")
print(train["Support Calls"].describe())

print("\nPayment Delay:")
print(train["Payment Delay"].describe())

print("\nTotal Spend:")
print(train["Total Spend"].describe())

print("\nLast Interaction:")
print(train["Last Interaction"].describe())

#Final data check
print("\n========== FINAL DATA CHECK ==========")

print("\nTraining shape:")
print(train.shape)

print("\nTesting shape:")
print(test.shape)

print("\nTraining missing values:")
print(train.isnull().sum().sum())

print("\nTesting missing values:")
print(test.isnull().sum().sum())

print("\nTraining duplicate rows:")
print(train.duplicated().sum())

print("\nTesting duplicate rows:")
print(test.duplicated().sum())

print("\nTraining duplicate Customer IDs:")
print(train["CustomerID"].duplicated().sum())

print("\nTesting duplicate Customer IDs:")
print(test["CustomerID"].duplicated().sum())

print("\nTraining data types:")
print(train.dtypes)

# Save cleaned datasets

train.to_csv("Customer Churn Dataset Training Cleaned.csv", index=False)
test.to_csv("Customer Churn Dataset Testing Cleaned.csv", index=False)

print("\nCleaned datasets saved successfully!")
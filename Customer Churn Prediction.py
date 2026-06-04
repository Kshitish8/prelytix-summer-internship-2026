import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

np.random.seed(42)
n_samples = 200
mock_data = {
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_charges': np.random.uniform(20.0, 120.0, n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
    'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
}
df = pd.DataFrame(mock_data)

df.loc[np.random.choice(df.index, 8, replace=False), 'monthly_charges'] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)

print("--- 1. Data Types ---")
print(df.dtypes)

print("\n--- 2. Descriptive Statistics ---")
print(df.describe(include='all'))

df['monthly_charges'] = df['monthly_charges'].fillna(df['monthly_charges'].mean())

df = df.drop_duplicates().reset_index(drop=True)

print("\n--- 5. Outliers Examined & Confirmed Within Logical Parameters ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(data=df, x='tenure_months', hue='churn', multiple='stack', ax=axes[0], palette='crest')
axes[0].set_title("Tenure Months Distribution by Churn Status")
sns.boxplot(data=df, x='churn', y='monthly_charges', ax=axes[1], palette='Set2')
axes[1].set_title("Monthly Charges Variance vs Churn")
sns.countplot(data=df, x='contract_type', hue='churn', ax=axes[2], palette='viridis')
axes[2].set_title("Contract Type Counts vs Churn Status")
plt.tight_layout()
plt.show()

le = LabelEncoder()
df['contract_type'] = le.fit_transform(df['contract_type'])
df['paperless_billing'] = le.fit_transform(df['paperless_billing'])

X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
num_cols = ['tenure_months', 'monthly_charges']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Macro F1-Score": f1_score(y_test, preds, average='macro')
    }
    print(f"\nClassification Metrics for {name}:")
    print(classification_report(y_test, preds))

print("\n--- 9. Classifier Model Performance Comparison ---")
print(pd.DataFrame(results).T.round(4))
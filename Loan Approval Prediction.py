import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score

np.random.seed(42)
n_samples = 300
mock_data = {
    'applicant_income': np.random.exponential(scale=5000, size=n_samples),
    'loan_amount': np.random.normal(loc=150, scale=50, size=n_samples),
    'credit_history_score': np.random.choice([1.0, 0.0], n_samples, p=[0.85, 0.15]),
    'loan_status': np.random.choice([1, 0], n_samples, p=[0.80, 0.20])
}
df = pd.DataFrame(mock_data)

df.loc[np.random.choice(df.index, 12, replace=False), 'credit_history_score'] = np.nan
df = pd.concat([df, df.iloc[:10]], ignore_index=True)

print("--- 1. Data Types ---")
print(df.dtypes)

print("\n--- 2. Descriptive Statistics ---")
print(df.describe())

df['credit_history_score'] = df['credit_history_score'].fillna(df['credit_history_score'].mode()[0])

df = df.drop_duplicates().reset_index(drop=True)

q_hi = df["applicant_income"].quantile(0.95)
df = df[df["applicant_income"] < q_hi].reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.countplot(data=df, x='loan_status', ax=axes[0], palette='Set1')
axes[0].set_title("Target Imbalance Breakdown Status (Loan Approved vs Rejected)")
sns.scatterplot(data=df, x='applicant_income', y='loan_amount', hue='loan_status', ax=axes[1], palette='coolwarm')
axes[1].set_title("Applicant Income vs Loan Amount Matrix")
sns.barplot(data=df, x='loan_status', y='credit_history_score', ax=axes[2], palette='muted')
axes[2].set_title("Credit History Availability Ratio vs Loan Status")
plt.tight_layout()
plt.show()

X = df.drop(columns=['loan_status'])
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

imb_classifiers = {
    "Weighted Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Weighted Decision Tree": DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=42),
    "Weighted Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
}

results = {}
for name, clf in imb_classifiers.items():
    clf.fit(X_train_scaled, y_train)
    preds = clf.predict(X_test_scaled)
    results[name] = {
        "Standard Accuracy": accuracy_score(y_test, preds),
        "Balanced Accuracy": balanced_accuracy_score(y_test, preds)
    }

print("\n--- 9. Balanced Loan Classifier Performance Comparison ---")
print(pd.DataFrame(results).T.round(4))
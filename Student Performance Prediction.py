import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report

np.random.seed(42)
n_samples = 250
mock_data = {
    'study_hours_weekly': np.random.uniform(2.0, 35.0, n_samples),
    'attendance_pct': np.random.uniform(60.0, 100.0, n_samples),
    'parental_support': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'final_score_marks': np.zeros(n_samples)
}
df = pd.DataFrame(mock_data)
df['final_score_marks'] = 20 + (df['study_hours_weekly'] * 1.2) + (df['attendance_pct'] * 0.4) + np.random.normal(0, 5, n_samples)
df['final_score_marks'] = np.clip(df['final_score_marks'], 0, 100)

df = pd.concat([df, df.iloc[:4]], ignore_index=True)

print("--- 1. Data Types ---")
print(df.dtypes)

print("\n--- 2. Descriptive Statistics ---")
print(df.describe(include='all'))

df = df.drop_duplicates().reset_index(drop=True)

pass_threshold = 50.0
df['pass_fail_label'] = df['final_score_marks'].apply(lambda x: 1 if x >= pass_threshold else 0)

X_features = df.drop(columns=['final_score_marks', 'pass_fail_label'])
y_regression = df['final_score_marks']
y_classification = df['pass_fail_label']

transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['study_hours_weekly', 'attendance_pct']),
        ('cat', OneHotEncoder(drop='first'), ['parental_support'])
    ])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df, x='study_hours_weekly', y='final_score_marks', hue='pass_fail_label', ax=axes[0], palette='bwr')
axes[0].set_title("Study Hours vs Final Score Marks")
sns.boxplot(data=df, x='parental_support', y='final_score_marks', ax=axes[1], order=['Low', 'Medium', 'High'], palette='pastel')
axes[1].set_title("Parental Support Quality vs Final Score")
sns.histplot(data=df, x='final_score_marks', kde=True, ax=axes[2], color='purple')
axes[2].axvline(pass_threshold, color='red', linestyle='--', label='Pass Limit Boundary')
axes[2].set_title("Score Density Population Distribution")
axes[2].legend()
plt.tight_layout()
plt.show()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_features, y_regression, test_size=0.2, random_state=42)
X_train_r_p = transformer.fit_transform(X_train_r)
X_test_r_p = transformer.transform(X_test_r)

reg_models = {
    "Ridge Regressor": Ridge(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}
reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_r_p, y_train_r)
    p = model.predict(X_test_r_p)
    reg_results[name] = {"RMSE": np.sqrt(mean_squared_error(y_test_r, p)), "R2": r2_score(y_test_r, p)}

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_features, y_classification, test_size=0.2, random_state=42, stratify=y_classification)
X_train_c_p = transformer.fit_transform(X_train_c)
X_test_c_p = transformer.transform(X_test_c)

clf_models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest Classifier": RandomForestClassifier(random_state=42)
}
clf_results = {}
for name, model in clf_models.items():
    model.fit(X_train_c_p, y_train_c)
    p = model.predict(X_test_c_p)
    clf_results[name] = {"Accuracy": accuracy_score(y_test_c, p)}

print("\n--- 9. Regression Models Evaluation Summary Tables ---")
print(pd.DataFrame(reg_results).T.round(4))

print("\n--- 9. Classification Models Evaluation Summary Tables ---")
print(pd.DataFrame(clf_results).T.round(4))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
n_samples = 200
mock_data = {
    'gr_liv_area': np.random.uniform(800, 3500, n_samples),
    'overall_qual': np.random.randint(1, 11, n_samples),
    'garage_cars': np.random.randint(0, 4, n_samples),
    'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples),
    'sale_price': np.zeros(n_samples)
}
df = pd.DataFrame(mock_data)
df['sale_price'] = 50000 + (df['gr_liv_area'] * 80) + (df['overall_qual'] * 15000) + np.random.normal(0, 15000, n_samples)

df.loc[np.random.choice(df.index, 10, replace=False), 'garage_cars'] = np.nan
df = pd.concat([df, df.iloc[:3]], ignore_index=True)
df.loc[50, 'gr_liv_area'] = 12000.0

print("--- 1. Data Types ---")
print(df.dtypes)

print("\n--- 2. Descriptive Statistics ---")
print(df.describe())

df['garage_cars'] = df['garage_cars'].fillna(df['garage_cars'].mode()[0])

df = df.drop_duplicates().reset_index(drop=True)

df = df[df['gr_liv_area'] < 6000].reset_index(drop=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.regplot(data=df, x='gr_liv_area', y='sale_price', ax=axes[0], color='crimson')
axes[0].set_title("Living Area vs Sale Price")
sns.boxplot(data=df, x='overall_qual', y='sale_price', ax=axes[1], palette='magma')
axes[1].set_title("Overall Quality Ranking vs Sale Price")
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2])
axes[2].set_title("Numeric Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

X = df.drop(columns=['sale_price'])
y = df['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['gr_liv_area', 'overall_qual', 'garage_cars']),
        ('cat', OneHotEncoder(drop='first'), ['neighborhood'])
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest Regressor": RandomForestRegressor(random_state=42)
}

results = {}
for name, reg in regressors.items():
    reg.fit(X_train_processed, y_train)
    preds = reg.predict(X_test_processed)
    results[name] = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2 Score": r2_score(y_test, preds)
    }

print("\n--- 9. Housing Model Performance Comparison ---")
print(pd.DataFrame(results).T.round(4))
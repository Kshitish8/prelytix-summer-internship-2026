import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
n_samples = 200
mock_data = {
    'trip_distance': np.random.uniform(0.5, 25.0, n_samples),
    'passenger_count': np.random.randint(1, 6, n_samples),
    'pickup_hour': np.random.randint(0, 24, n_samples),
    'trip_duration_mins': np.random.uniform(5.0, 90.0, n_samples),
    'fare_amount': np.zeros(n_samples)
}
df = pd.DataFrame(mock_data)
df['fare_amount'] = 2.5 + (df['trip_distance'] * 3.0) + (df['trip_duration_mins'] * 0.5) + np.random.normal(0, 2, n_samples)

df.loc[np.random.choice(df.index, 10, replace=False), 'trip_duration_mins'] = np.nan
df.loc[np.random.choice(df.index, 5, replace=False), 'fare_amount'] = np.nan
df = pd.concat([df, df.iloc[:5]], ignore_index=True)
df.loc[15, 'trip_distance'] = 350.0
df.loc[32, 'fare_amount'] = 900.0

print("--- 1. Data Types ---")
print(df.dtypes)

print("\n--- 2. Descriptive Statistics ---")
print(df.describe())

print("\n--- 3. Missing Values Pre-Imputation ---")
print(df.isnull().sum())
df['trip_duration_mins'] = df['trip_duration_mins'].fillna(df['trip_duration_mins'].median())
df['fare_amount'] = df['fare_amount'].fillna(df['fare_amount'].median())

print(f"\n--- 4. Duplicates Detected: {df.duplicated().sum()} rows ---")
df = df.drop_duplicates().reset_index(drop=True)

for col in ['trip_distance', 'fare_amount']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)
print("\n--- 5. Outliers Handled via IQR Capping ---")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(data=df, x='trip_distance', y='fare_amount', ax=axes[0], color='tab:blue')
axes[0].set_title("Distance vs Fare Amount")
sns.histplot(df['fare_amount'], kde=True, ax=axes[1], color='tab:green')
axes[1].set_title("Distribution of Fare Amount")
sns.boxplot(data=df, x='passenger_count', y='fare_amount', ax=axes[2], color='tab:orange')
axes[2].set_title("Passenger Count vs Fare Amount")
plt.tight_layout()
plt.show()

df['is_rush_hour'] = df['pickup_hour'].apply(lambda x: 1 if (8<=x<=10 or 17<=x<=19) else 0)

X = df.drop(columns=['fare_amount'])
y = df['fare_amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2 Score": r2}

print("\n--- 9. Model Performance Comparison ---")
print(pd.DataFrame(results).T.round(4))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('cleaned_houses_Madrid.csv')
print("Data shape before:", df.shape)

columns_to_remove = ['street_name', 'street_number', 'is_floor_under',
                      'operation', 'is_new_development'
]
                      
df = df.drop(columns=columns_to_remove, errors='ignore')

df = pd.get_dummies(df, columns=['neighborhood_id', 'house_type_id'], dtype=int)

print("Data shape after:", df.shape)

# df.to_csv('ml_houses_Madrid_encoded.csv', index=False)

# --------------------------

y = df['buy_price']
X = df.drop(columns='buy_price')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------

print("Training model: LinearRegression")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"MAE error: {mae_lr:,.2f} euro")
print(f"RMSE error: {rmse_lr:,.2f} euro")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


df = pd.read_csv('cleaned_houses_Madrid.csv')

print("--------------------------")

print("Data shape before:", df.shape)

columns_to_remove = ['street_name', 'street_number', 'is_floor_under',
                      'operation', 'is_new_development', 'buy_price_by_area', 'is_buy_price_known'
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

print("--------------------------")

print("Training model: LinearRegression")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print(f"MAE error: {mae_lr:,.2f} euro")
print(f"RMSE error: {rmse_lr:,.2f} euro")

print("--------------------------")

# --------------------------

print("Training model: RandomForest")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)

rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print(f"MAE error: {mae_rf:,.2f} euro")
print(f"RMSE error: {rmse_rf:,.2f} euro")

# --------------------------

print("--------------------------")

ann_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    random_state=42
)

print("Training model: ANN")

ann_model.fit(X_train_scaled, y_train)

y_pred_ann = ann_model.predict(X_test_scaled)

mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))

print(f"MAE error: {mae_ann:,.2f} euro")
print(f"RMSE error: {rmse_ann:,.2f} euro")

print("--------------------------")

# --------------------------

ann_model_2 = MLPRegressor(
    hidden_layer_sizes=(256,),
    activation='relu',
    solver='adam',
    max_iter=2000,
    random_state=42
)

print("Training model: SNN")

ann_model_2.fit(X_train_scaled, y_train)

y_pred_ann_2 = ann_model_2.predict(X_test_scaled)

mae_ann_2 = mean_absolute_error(y_test, y_pred_ann_2)
rmse_ann_2 = np.sqrt(mean_squared_error(y_test, y_pred_ann_2))

print(f"MAE error: {mae_ann_2:,.2f} euro")
print(f"RMSE error: {rmse_ann_2:,.2f} euro")
import pandas as pd

df = pd.read_csv('cleaned_houses_Madrid.csv')
print("Data shape before:", df.shape)

columns_to_remove = ['street_name', 'street_number', 'is_floor_under',
                      'operation', 'is_new_development', 'rent_price', 'is_rent_price_known']
                      
df = df.drop(columns=columns_to_remove, errors='ignore')

df = pd.get_dummies(df, columns=['neighborhood_id', 'house_type_id'], dtype=int)

print("Data shape after:", df.shape)

df.to_csv('ml_houses_Madrid_encoded.csv', index=False)
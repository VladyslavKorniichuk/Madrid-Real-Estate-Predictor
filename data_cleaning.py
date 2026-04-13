import pandas as pd

df = pd.read_csv("houses_Madrid.csv")
print("Data shape before:", df.shape)

columns_to_remove = [ # columns which are unnecessary
    'id', 'title', 'subtitle', 'sq_mt_allotment', 'raw_address',
    'is_exact_adrress_hidden', 'portal', 'sq_mt_useful', 'door',
    'external_reference', 'energy_certificate', 'n_floors'
]

df = df.drop(columns=columns_to_remove, errors='ignore')    # remove columns which are include in columns_to_remove variable
df = df.dropna(subset=['buy_price'])                        # remove data, where buy_price is N/A
df = df.dropna(axis=1, how='all')                           # remove the columns, which are completely empty

df['sq_mt_built'] = df['sq_mt_built'].fillna(df['sq_mt_built'].median())
df['n_rooms'] = df['n_rooms'].fillna(df['n_rooms'].median())          
df['n_bathrooms'] = df['n_bathrooms'].fillna(df['n_bathrooms'].median())

if 'has_elevator' in df.columns:
    df['has_elevator'] = df['has_elevator'].astype(int) # replacing bool to int 
if 'has_parking' in df.columns:
    df['has_parking'] = df['has_parking'].astype(int)

print("Data shape after:", df.shape)

print(df.info())
df.to_csv('cleaned_houses_Madrid.csv', index=False)
import pandas as pd

df = pd.read_csv("houses_Madrid.csv")
print("Data shape before:", df.shape)

columns_to_remove = [                                       # columns which are unnecessary
    'id', 'title', 'subtitle', 'sq_mt_allotment', 'raw_address',
    'is_exact_address_hidden', 'portal', 'sq_mt_useful', 'door',
    'external_reference', 'energy_certificate', 'n_floors'
]

df = df.drop(columns=columns_to_remove, errors='ignore')    # remove columns which are include in columns_to_remove variable
df = df.dropna(subset=['buy_price'])                        # remove data, where buy_price is N/A
df = df.dropna(axis=1, how='all')                           # remove the columns, which are completely empty

df['sq_mt_built'] = df['sq_mt_built'].fillna(df['sq_mt_built'].median())        # fill blank values with median value
df['n_rooms'] = df['n_rooms'].fillna(df['n_rooms'].median())                    # fill blank values with median value
df['n_bathrooms'] = df['n_bathrooms'].fillna(df['n_bathrooms'].median())        # fill blank values with median value

for col in df.columns:
    if col.startswith('is_') or col.startswith('has_'):
        df[col] = df[col].fillna(0)                                             # if columns is empty, the fill it with 0 value
        df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0})   # replace word/value for 1/0
        df[col] = df[col].astype(int)                                           # convert type to int

print("Data shape after:", df.shape)

print(df.info())
df.to_csv('cleaned_houses_Madrid.csv', index=False)

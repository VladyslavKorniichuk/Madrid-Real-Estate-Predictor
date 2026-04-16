import pandas as pd

df = pd.read_csv("houses_Madrid.csv")
print("Data shape before:", df.shape)

# Drop features which are unnecessary for price prediction 
columns_to_remove = [
    'id', 'title', 'subtitle', 'sq_mt_allotment', 'raw_address',
    'is_exact_address_hidden', 'portal', 'sq_mt_useful', 'door',
    'external_reference', 'energy_certificate', 'n_floors'
]
df = df.drop(columns=columns_to_remove, errors='ignore')

# Clean rows and columns missing data
df = df.dropna(subset=['buy_price'])  # Target variable cannot be null
df = df.dropna(axis=1, how='all')     # Drop empty columns

floor_swap = {
    'Bajo': '0', 
    'Entreplanta': '0', 
    'Semi-sÃ³tano': '-1',
    'Semi-sótano': '-1'
}

df['floor'] = df['floor'].replace(floor_swap)
df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
df['floor'] = df['floor'].fillna(df['floor'].median())
df['floor'] = df['floor'].astype(int)

# Impute missing numerical values using median (to avoid influence of extreme outliers)
df['sq_mt_built'] = df['sq_mt_built'].fillna(df['sq_mt_built'].median())
df['n_rooms'] = df['n_rooms'].fillna(df['n_rooms'].median())
df['n_bathrooms'] = df['n_bathrooms'].fillna(df['n_bathrooms'].median())

# Standardize boolean columns into binary integer format (0/1) for modeling
for col in df.columns:
    if col.startswith('is_') or col.startswith('has_'):
        df[col] = df[col].fillna(0)
        df[col] = df[col].replace({True: 1, False: 0, 'True': 1, 'False': 0})
        df[col] = df[col].astype(int)

print("Data shape after:", df.shape)

df.to_csv('cleaned_houses_Madrid.csv', index=False)
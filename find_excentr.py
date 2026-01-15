import pandas as pd

# 1. Charger le fichier
FILE_NAME = "auteurs_avec_excentricite_et_domaine.csv"
print("Chargement du fichier...")
df = pd.read_csv(FILE_NAME)

# 2. Correction des degrés (diviser par 2 car doublés par le script GPU)
df['Degré'] = df['Degré'] / 2

# 3. FILTRER : On garde seulement ceux qui ont une Excentricité > 0
# (Les non-calculés sont soit NaN (vide), soit 0.0)
df_ecc = df[ (df['Excentricité'].notna()) & (df['Excentricité'] > 0) ]

# 4. TRIER :
# - Les plus petits chiffres = Les auteurs les plus CENTRAUX (proches de tout le monde)
# - Les plus grands chiffres = Les auteurs PÉRIPHÉRIQUES
df_ecc = df_ecc.sort_values(by='Excentricité', ascending=True)

# 5. Afficher et Sauvegarder
print(f"Nombre d'auteurs avec excentricité calculée : {len(df_ecc)}")
print("\n--- Top 10 des auteurs les plus centraux (Excentricité faible) ---")
print(df_ecc[['Auteur', 'Excentricité', 'Degré']].head(10))

print("\n--- Top 10 des auteurs les plus périphériques (Excentricité élevée) ---")
print(df_ecc[['Auteur', 'Excentricité', 'Degré']].tail(10))

# Sauvegarder ce sous-ensemble dans un nouveau fichier pour analyse
df_ecc.to_csv("auteurs_avec_excentricite_filtree_et_domaine.csv", index=False)
print("\nFichier filtré sauvegardé sous : 'auteurs_avec_excentricite.csv'")
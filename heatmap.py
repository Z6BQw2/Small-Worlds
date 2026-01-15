import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import chi2_contingency
import numpy as np

# 1. Chargement des données existantes
try:
    df = pd.read_csv("auteurs_avec_excentricite_et_domaine.csv")
except FileNotFoundError:
    # Essayer le sous-dossier si besoin
    df = pd.read_csv("Fichiers_finaux/auteurs_avec_excentricite_et_domaine.csv")

df = df.fillna(0)

# 2. Reconstitution des Clusters (Car ils ne sont pas dans le CSV)
features = ['Degré', 'Clustering', 'Centralité Intermédiarité', 'Excentricité']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Nommage des Clusters
means = df.groupby('Cluster')[features].mean()
hub_id = means['Degré'].idxmax()
remaining = list(set(means.index) - {hub_id})
peri_id = means.loc[remaining, 'Excentricité'].idxmax()
comm_id = list(set(remaining) - {peri_id})[0]
cluster_names = {hub_id: 'Hubs', peri_id: 'Périphériques', comm_id: 'Communautaires'}
df['Cluster_Label'] = df['Cluster'].map(cluster_names)

# 3. Création des "Domaines Généraux" (Macro)
def map_to_general(domaine):
    if not isinstance(domaine, str): return 'Autre'
    d = domaine.strip()
    if any(x in d for x in ['IA', 'Machine Learning', 'Vision', 'Langage', 'Neurones', 'Robotique', 'Stats & ML', 'Informatique', 'Signal', 'cs.']): return 'Informatique'
    if any(x in d for x in ['Physique', 'Matière', 'Astro', 'Quantique', 'Physics']): return 'Physique'
    if any(x in d for x in ['Maths', 'Statistiques', 'Proba']): return 'Mathématiques'
    if any(x in d for x in ['Biologie', 'Neuro']): return 'Biologie'
    return 'Autre'

df['Domaine_General'] = df['Domaine_Dominant'].apply(map_to_general)
df_clean = df[~df['Domaine_General'].isin(['Autre', 'Inconnu'])]

# 4. Génération de la Heatmap
contingency_table = pd.crosstab(df_clean['Cluster_Label'], df_clean['Domaine_General'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
residuals = (contingency_table - expected) / np.sqrt(expected)

plt.figure(figsize=(10, 6))
sns.heatmap(residuals, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-4, vmax=4)
plt.title(f"Heatmap des Résidus (Domaines Généraux)\np-value = {p:.2e}")
plt.ylabel("Profil Structurel")
plt.xlabel("Grand Domaine")
plt.tight_layout()
plt.savefig("Heatmap_Generale.png")
print(f"Image générée : Heatmap_Generale.png (p-value={p})")
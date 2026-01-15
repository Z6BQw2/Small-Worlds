import cugraph
import cudf
import pandas as pd
import json
import time
from tqdm import tqdm
import random
import numpy as np

# --- CONFIGURATION ---
JSON_FILE = "graphe_bengio_network__clean_Copie_.jsonl"
CSV_FILE = "resultats_finaux_gpu_optimized.csv"      # Le fichier source (avec degrés)
FINAL_FILE = "resultats_finaux_distrib_represent.csv" # Le fichier de sortie
SAMPLE_SIZE = 50000 # 1000 est suffisant pour une marge d'erreur ~3%

def run_representative_sampling():
    print(f"[{time.strftime('%H:%M:%S')}] Chargement des données...")
    
    # 1. Charger le CSV des métriques existantes
    try:
        df_csv = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print("Erreur: Fichier CSV introuvable.")
        return

    # CORRECTION DES DEGRÉS (si ce n'est pas déjà fait)
    # On vérifie si c'est pair en moyenne pour deviner s'il faut diviser
    if df_csv['Degré'].mean() > 10 and df_csv['Degré'].iloc[0] % 2 == 0:
        print("Correction des degrés (Division par 2)...")
        df_csv['Degré'] = df_csv['Degré'] / 2

    # 2. Reconstruire le Graphe pour trouver la LCC
    print(f"[{time.strftime('%H:%M:%S')}] Reconstruction du graphe (pour LCC)...")
    edges = []
    nodes_seen = set()
    
    # Lecture optimisée juste pour les arêtes
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                src = data.get("author")
                coauthors = data.get("coauthors", [])
                if src:
                    nodes_seen.add(src)
                    for dst in coauthors:
                        if src != dst:
                            edges.append([src, dst])
                            nodes_seen.add(dst)
            except: continue
            
    # Mapping
    all_nodes = list(nodes_seen)
    node_to_id = {name: i for i, name in enumerate(all_nodes)}
    id_to_node = {i: name for i, name in enumerate(all_nodes)}
    
    pdf_edges = pd.DataFrame(edges, columns=['src', 'dst'])
    pdf_edges['src_id'] = pdf_edges['src'].map(node_to_id)
    pdf_edges['dst_id'] = pdf_edges['dst'].map(node_to_id)
    pdf_edges = pdf_edges.dropna().astype({'src_id': 'int32', 'dst_id': 'int32'})
    
    # GPU Load
    gdf_edges = cudf.DataFrame.from_pandas(pdf_edges[['src_id', 'dst_id']])
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(gdf_edges, source='src_id', destination='dst_id', renumber=False)

    # 3. IDENTIFICATION DE LA POPULATION CIBLE (LCC)
    print(f"[{time.strftime('%H:%M:%S')}] Isolation de la Composante Connexe Géante (LCC)...")
    components = cugraph.connected_components(G)
    largest_label = components['labels'].value_counts().index[0]
    
    # Récupérer les IDs GPU des nœuds de la LCC
    lcc_gpu_nodes = components[components['labels'] == largest_label]['vertex']
    
    # Convertir en liste Python d'IDs
    valid_ids = set(lcc_gpu_nodes.to_pandas().tolist())
    
    # Retrouver les noms d'auteurs correspondant à ces IDs
    # C'est important : on ne peut échantillonner que parmi les gens DANS la LCC
    valid_authors = [id_to_node[i] for i in valid_ids if i in id_to_node]
    
    print(f"Population LCC : {len(valid_authors)} auteurs ({len(valid_authors)/len(all_nodes):.1%} du graphe total).")

    # 4. ÉCHANTILLONNAGE REPRÉSENTATIF
    # On filtre notre DataFrame CSV pour ne garder que ceux qui sont dans la LCC
    df_population = df_csv[df_csv['Auteur'].isin(valid_authors)]
    
    print(f"[{time.strftime('%H:%M:%S')}] Tirage aléatoire de {SAMPLE_SIZE} auteurs...")
    # L'échantillonnage aléatoire simple préserve la distribution des degrés
    df_sample = df_population.sample(n=min(SAMPLE_SIZE, len(df_population)), random_state=42)
    
    # Vérification de la distribution (Bonus)
    mean_pop = df_population['Degré'].mean()
    mean_samp = df_sample['Degré'].mean()
    print(f"   -> Degré moyen Population : {mean_pop:.2f}")
    print(f"   -> Degré moyen Échantillon: {mean_samp:.2f}")
    print("   (Si les chiffres sont proches, l'échantillon est représentatif)")

    # Préparer les IDs pour le calcul
    target_ids = [node_to_id[name] for name in df_sample['Auteur'].tolist()]

    # 5. CALCUL GPU (BFS)
    # Préparer le sous-graphe LCC sur GPU
    gdf_lcc = gdf_edges[gdf_edges.src_id.isin(lcc_gpu_nodes) & gdf_edges.dst_id.isin(lcc_gpu_nodes)]
    G_lcc = cugraph.Graph(directed=False)
    G_lcc.from_cudf_edgelist(gdf_lcc, source='src_id', destination='dst_id', renumber=False)
    
    print(f"[{time.strftime('%H:%M:%S')}] Calcul de l'excentricité (BFS) pour l'échantillon...")
    results = []
    
    for node_id in tqdm(target_ids):
        try:
            # BFS pour distance max
            df_paths = cugraph.bfs(G_lcc, start=node_id)
            max_dist = df_paths['distance'].max()
            
            if max_dist < 1e9 and max_dist > 0:
                results.append({'vertex': node_id, 'Excentricité_Rep': max_dist})
        except: pass

    # 6. SAUVEGARDE
    if results:
        d_ecc = pd.DataFrame(results)
        d_ecc['Auteur'] = d_ecc['vertex'].map(id_to_node)
        
        # On repart du fichier CSV complet original
        # On nettoie l'ancienne colonne excentricité si elle existe
        if 'Excentricité' in df_csv.columns:
            del df_csv['Excentricité']
            
        # On fusionne les nouvelles données
        df_final = df_csv.merge(d_ecc[['Auteur', 'Excentricité_Rep']], on='Auteur', how='left')
        df_final.rename(columns={'Excentricité_Rep': 'Excentricité'}, inplace=True)
        
        df_final.to_csv(FINAL_FILE, index=False)
        print(f"\n✅ TERMINÉ. Fichier représentatif généré : {FINAL_FILE}")
        
        # Stats finales
        print(df_final['Excentricité'].describe())
    else:
        print("Erreur : Aucun calcul n'a réussi.")

if __name__ == "__main__":
    run_representative_sampling()
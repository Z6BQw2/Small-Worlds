import networkx as nx
import pandas as pd
import json
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION RAPIDE ---
JSON_FILE = "graphe_bengio_network__clean_Copie_.jsonl"
CSV_FILE = "auteurs_avec_excentricite_filtree_et_domaine.csv"
SAMPLE_SIZE = 15 # Nombre de noeuds pour estimer L (pour aller vite)

def load_data():
    print("1. Chargement des données...")
    # Charger les Domaines
    df = pd.read_csv(CSV_FILE)
    # Dictionnaire {Auteur: Domaine}
    domains = df.set_index('Auteur')['Domaine_Dominant'].to_dict()
    
    # Charger le Graphe
    G = nx.Graph()
    with open(JSON_FILE, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                src = data.get("author")
                coauthors = data.get("coauthors", [])
                if src:
                    # On note le domaine du noeud
                    G.add_node(src, domain=domains.get(src, "Inconnu"))
                    for dst in coauthors:
                        if dst != src:
                            G.add_node(dst, domain=domains.get(dst, "Inconnu"))
                            G.add_edge(src, dst)
            except: continue
    return G

def get_sampled_average_path_length(G, n_samples=50):
    """ Calcule L approximatif sur la plus grande composante connexe """
    if len(G) == 0: return 0
    
    # On travaille uniquement sur la composante géante (LCC)
    largest_cc = max(nx.connected_components(G), key=len)
    subG = G.subgraph(largest_cc)
    
    nodes = list(subG.nodes())
    # Si le graphe est petit, on prend tout, sinon on échantillonne
    sources = random.sample(nodes, min(n_samples, len(nodes)))
    
    total_path_lengths = 0
    count = 0
    
    for source in sources:
        lengths = nx.single_source_shortest_path_length(subG, source)
        # On ne garde que les distances > 0
        total_path_lengths += sum(lengths.values())
        count += len(lengths) - 1 # -1 pour ne pas compter soi-même
        
    return total_path_lengths / count if count > 0 else 0

def run_attack():
    G = load_data()
    print(f"Graphe initial : {len(G.nodes())} noeuds, {len(G.edges())} liens.")
    
    # 2. Classification des Liens
    inter_edges = []
    intra_edges = []
    
    print("Classification des arêtes...")
    for u, v in G.edges():
        dom_u = G.nodes[u].get('domain', 'Inconnu')
        dom_v = G.nodes[v].get('domain', 'Inconnu')
        
        if dom_u != "Inconnu" and dom_v != "Inconnu" and dom_u != dom_v:
            inter_edges.append((u, v))
        else:
            intra_edges.append((u, v))
            
    print(f"-> Liens Inter-Domaines (Ponts) : {len(inter_edges)}")
    print(f"-> Liens Intra-Domaines (Communautés) : {len(intra_edges)}")

    # 3. Simulation
    results = {'Inter': [], 'Intra': []}
    percentages = [0, 0.05, 0.10, 0.15]
# On coupe 0%, 10%, 30%, 50%
    
    initial_L = get_sampled_average_path_length(G, SAMPLE_SIZE)
    print(f"Distance Moyenne (L) Initiale : {initial_L:.2f}")
    
    # --- Attaque INTER ---
    print("\n--- Attaque INTER-DOMAINES ---")
    for p in percentages:
        G_temp = G.copy()
        n_remove = int(len(inter_edges) * p)
        if n_remove > 0:
            edges_to_remove = random.sample(inter_edges, n_remove)
            G_temp.remove_edges_from(edges_to_remove)
        
        L = get_sampled_average_path_length(G_temp, SAMPLE_SIZE)
        results['Inter'].append(L)
        print(f"Coupe {int(p*100)}% : L = {L:.2f}")

    # --- Attaque INTRA ---
    print("\n--- Attaque INTRA-DOMAINES ---")
    for p in percentages:
        G_temp = G.copy()
        n_remove = int(len(intra_edges) * p) # On enlève le MÊME NOMBRE de liens que pour inter pour être fair-play ? 
        # NON, on enlève le même pourcentage de la catégorie pour tester la robustesse structurelle
        if n_remove > 0:
            edges_to_remove = random.sample(intra_edges, n_remove)
            G_temp.remove_edges_from(edges_to_remove)
            
        L = get_sampled_average_path_length(G_temp, SAMPLE_SIZE)
        results['Intra'].append(L)
        print(f"Coupe {int(p*100)}% : L = {L:.2f}")

    # 4. Plot Rapide
    plt.figure(figsize=(8, 5))
    plt.plot(percentages, results['Inter'], 'r-o', label='Coupe Inter-Domaines (Ponts)', linewidth=3)
    plt.plot(percentages, results['Intra'], 'g--o', label='Coupe Intra-Domaines (Communautés)')
    plt.title("Impact de la suppression des liens sur la Distance Moyenne (L)")
    plt.xlabel("% de liens supprimés")
    plt.ylabel("Distance Moyenne (L)")
    plt.legend()
    plt.grid(True)
    plt.show()

run_attack()
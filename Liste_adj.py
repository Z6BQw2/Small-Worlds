# On garde les imports de vos coéquipiers
import arxiv
import time
import pandas as pd
from collections import deque
import json # Pour écrire au format JSON
from tqdm import tqdm # Pour la barre de progression

# --- La fonction get_coauthors_from_arxiv reste la même ---
def get_coauthors_from_arxiv(author_name: str) -> set[str]:
    try:
        # NOTE: La librairie arxiv gère déjà une attente pour respecter l'API.
        # Pour des tests rapides, on peut la rendre plus agressive, mais
        # attention à ne pas se faire bannir.
        client = arxiv.Client(
            page_size=50,
            delay_seconds=1.0, # On réduit un peu pour l'exemple, 3.0 est plus sûr
            num_retries=5
        )
        search = arxiv.Search(
            query=f'au:"{author_name}"',
            max_results=100, # On peut limiter le nombre de papiers par auteur
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = client.results(search)
        all_coauthors = set()
        for r in results:
            authors_in_paper = {auth.name for auth in r.authors}
            all_coauthors.update(authors_in_paper)
        all_coauthors.discard(author_name)
        return all_coauthors
    except Exception as e:
        print(f"Erreur lors de la recherche pour '{author_name}': {e}")
        return set()

# --- PHASE 1: Construction du graphe avec écriture en continu ---

def stream_build_graph_to_jsonl(seed_author: str, output_file: str): # Plus besoin de max_authors
    """
    Construit le graphe en parcourant les auteurs (BFS) et écrit chaque
    relation découverte dans un fichier JSON Lines en temps réel.
    """
    print(f"--- Démarrage de la collecte. Les données seront sauvées dans '{output_file}' ---")
    
    queue = deque([seed_author])
    
    # On doit charger les auteurs déjà visités si on reprend une exploration
    visited = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                visited.add(record['author'])
        print(f"Reprise de l'exploration avec {len(visited)} auteurs déjà dans le fichier.")
        # On doit aussi re-remplir la queue avec des auteurs non explorés
        # C'est complexe, la solution simple est de repartir de la graine
        if seed_author not in visited:
            visited.add(seed_author)
        else: # Pour éviter d'explorer les mêmes
             print("Attention : le point de départ a déjà été exploré.")

    except FileNotFoundError:
        visited.add(seed_author)

    with open(output_file, 'a', encoding='utf-8') as f:
        # tqdm sans 'total' devient une barre de progression infinie
        pbar = tqdm(desc="Auteurs explorés")
        
        # --- LA CONDITION SIMPLIFIÉE ---
        while queue:
            current_author = queue.popleft()
            pbar.set_description(f"Exploration de {current_author}")

            coauthors = get_coauthors_from_arxiv(current_author)
            
            # On écrit les données de l'auteur courant (même s'il était déjà visité)
            record = {'author': current_author, 'coauthors': list(coauthors)}
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            for coauthor in coauthors:
                if coauthor not in visited:
                    visited.add(coauthor)
                    queue.append(coauthor)
            
            pbar.update(1)
            
        pbar.close()

    print(f"\n--- Exploration terminée (la composante connexe a été entièrement parcourue). {len(visited)} auteurs découverts. ---")

# --- PHASE 2: Chargement du graphe et extraction des features ---

def load_graph_from_jsonl(input_file: str) -> dict[str, set]:
    """
    Lit un fichier JSON Lines et reconstruit le graphe (liste d'adjacence) en mémoire.
    """
    print(f"--- Chargement du graphe depuis '{input_file}' ---")
    graph = {}
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    author = record['author']
                    coauthors = set(record['coauthors'])
                    
                    # Assure que l'auteur a une entrée dans le graphe
                    if author not in graph:
                        graph[author] = set()
                    
                    # Ajoute les liens sortants
                    graph[author].update(coauthors)
                    
                    # Ajoute les liens entrants pour un graphe non-dirigé
                    for coauthor in coauthors:
                        if coauthor not in graph:
                            graph[coauthor] = set()
                        graph[coauthor].add(author)
                except json.JSONDecodeError:
                    print(f"Attention: Ligne malformée ignorée dans {input_file}")
                    
    except FileNotFoundError:
        print(f"Erreur: Le fichier {input_file} n'a pas été trouvé.")
        return {}
        
    print(f"--- Graphe chargé avec {len(graph)} auteurs. ---")
    return graph

# Les fonctions calculate_local_clustering_coefficient et extract_features_for_pca
# restent EXACTEMENT les mêmes. Elles travaillent sur un graphe déjà construit.
def calculate_local_clustering_coefficient(author: str, graph: dict) -> float:
    neighbors = list(graph.get(author, set()))
    k = len(neighbors)
    if k < 2: return 0.0
    possible_links = k * (k - 1) / 2
    existing_links = 0
    for i in range(k):
        for j in range(i + 1, k):
            if neighbors[j] in graph.get(neighbors[i], set()):
                existing_links += 1
    return existing_links / possible_links if possible_links > 0 else 0.0

def extract_features_for_pca(graph: dict) -> pd.DataFrame:
    print("--- Démarrage de l'extraction des features ---")
    features = []
    for author in tqdm(graph, desc="Calcul des features"):
        neighbors = graph[author]
        degree = len(neighbors)
        local_clustering = calculate_local_clustering_coefficient(author, graph)
        second_degree_neighbors = set()
        for neighbor in neighbors:
            second_degree_neighbors.update(graph.get(neighbor, set()))
        second_degree_neighbors.discard(author)
        second_degree_neighbors -= neighbors
        num_second_degree = len(second_degree_neighbors)
        features.append({
            'author': author,
            'degree': degree,
            'clustering_coeff': local_clustering,
            'second_degree_neighbors': num_second_degree
        })
    print("--- Extraction terminée. ---")
    return pd.DataFrame(features).set_index('author')

# --- Script Principal ---

if __name__ == "__main__":
    SEED_AUTHOR = "Yoshua Bengio"
    # Fichier de sortie pour les données brutes
    GRAPH_DATA_FILE = "graphe_bengio_network.jsonl"
    # Limite pour l'exploration. Vous pouvez l'augmenter.
    MAX_AUTHORS_TO_EXPLORE = 2000
    
    # --- PHASE 1 ---
    # Cette fonction va tourner longtemps. Vous pouvez l'arrêter avec Ctrl+C.
    # Le fichier GRAPH_DATA_FILE contiendra votre progression.
    # Si vous voulez recommencer de zéro, supprimez ce fichier.
    stream_build_graph_to_jsonl(
        seed_author=SEED_AUTHOR,
        output_file=GRAPH_DATA_FILE,
    )
    
    # --- PHASE 2 ---
    # Une fois la collecte terminée (ou arrêtée), on peut analyser les données.
    # Cette partie est rapide car elle ne fait pas d'appels réseau.
    collaboration_graph = load_graph_from_jsonl(GRAPH_DATA_FILE)
    
    # On continue uniquement si le graphe n'est pas vide
    if collaboration_graph:
        features_df = extract_features_for_pca(collaboration_graph)
        
        print("\nDataFrame des features prêtes pour la PCA:")
        print(features_df.head())
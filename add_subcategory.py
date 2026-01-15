import pandas as pd
import json
import arxiv
import time
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
# Fichiers d'entrée
JSON_GRAPH_FILE = "graphe_bengio_network__clean_Copie_.jsonl"
CSV_METRICS_FILE = "resultats_complets_avec_categories.csv" # Ton CSV avec Degré, Clustering, etc.

# Fichier de sortie
OUTPUT_CSV_FILE = "auteurs_avec_excentricite_et_domaine.csv" # Le nouveau fichier enrichi

# Mapping pour regrouper les catégories d'ArXiv en grands domaines
# C'est ici que tu peux affiner pour avoir une meilleure granularité
DOMAIN_MAP = {
    'cs.AI': 'IA',
    'cs.LG': 'Machine Learning',
    'cs.CV': 'Vision par Ordinateur',
    'cs.CL': 'Traitement du Langage',
    'cs.NE': 'Réseaux de Neurones',
    'cs.RO': 'Robotique',
    'stat.ML': 'Stats & ML',
    'quant-ph': 'Physique Quantique',
    'math.PR': 'Maths (Proba)',
    'math.ST': 'Maths (Stats)',
    'cond-mat': 'Physique (Matière Cond.)',
    'astro-ph': 'Physique (Astro)',
    'eess.SP': 'Ingénierie (Signal)',
    'econ.GN': 'Économie',
    'q-bio.NC': 'Biologie (Neuro)',
}

def get_specific_domain(arxiv_category):
    """
    Tente de trouver une catégorie spécifique dans notre map.
    Sinon, renvoie la catégorie principale (ex: 'cs').
    """
    if not arxiv_category or not isinstance(arxiv_category, str):
        return "Inconnu"
    # Correspondance exacte prioritaire
    if arxiv_category in DOMAIN_MAP:
        return DOMAIN_MAP[arxiv_category]
    # Correspondance par préfixe
    prefix = arxiv_category.split('.')[0]
    if prefix == 'cs': return 'Informatique (Autre)'
    if prefix == 'math': return 'Maths (Autre)'
    if prefix == 'stat': return 'Statistiques (Autre)'
    if prefix in ['physics', 'phys', 'quant-ph', 'cond-mat', 'astro-ph', 'hep']:
        return 'Physique (Autre)'
    return "Autre"

def fetch_domains_from_api(authors_list):
    """
    Interroge l'API ArXiv pour une liste d'auteurs et renvoie leur domaine principal.
    C'est l'étape la plus lente.
    """
    author_domains = {}
    print(f"--- Interrogation de l'API ArXiv pour {len(authors_list)} auteurs ---")
    
    client = arxiv.Client(page_size=1, delay_seconds=1.0, num_retries=3)
    
    for author in tqdm(authors_list, desc="API ArXiv"):
        try:
            search = arxiv.Search(query=f'au:"{author}"', max_results=1)
            # Utiliser un try-except pour les auteurs sans publication
            paper = next(client.results(search), None)
            if paper:
                category = paper.primary_category
                author_domains[author] = get_specific_domain(category)
            else:
                author_domains[author] = "Inconnu (API)"
        except Exception as e:
            print(f"Erreur API pour l'auteur '{author}': {e}")
            author_domains[author] = "Erreur API"
            
    return author_domains

def main():
    # 1. Lire le CSV de métriques pour savoir quels auteurs nous intéressent
    try:
        df_metrics = pd.read_csv(CSV_METRICS_FILE)
    except FileNotFoundError:
        print(f"ERREUR: Le fichier '{CSV_METRICS_FILE}' est introuvable.")
        return
    
    target_authors = set(df_metrics['Auteur'])
    print(f"{len(target_authors)} auteurs cibles chargés depuis '{CSV_METRICS_FILE}'.")

    # 2. Lire le graphe JSONL pour identifier les 'seeds' et construire la structure de voisinage
    seeds = set()
    # Structure : {co-auteur: [liste des seeds auxquels il est connecté]}
    reverse_adjacency = {}
    
    print("Analyse de la structure du graphe depuis le fichier JSONL...")
    with open(JSON_GRAPH_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            seed_author = data.get("author")
            if seed_author:
                seeds.add(seed_author)
                for coauthor in data.get("coauthors", []):
                    if coauthor not in reverse_adjacency:
                        reverse_adjacency[coauthor] = []
                    reverse_adjacency[coauthor].append(seed_author)

    print(f"{len(seeds)} auteurs 'sources' (seeds) identifiés.")

    # 3. Récupérer les domaines des seeds via l'API
    seed_domains = fetch_domains_from_api(list(seeds))

    # 4. Propager les domaines aux autres auteurs par vote majoritaire
    final_author_domains = {}
    print("\nPropagation des domaines par vote majoritaire...")
    
    for author in tqdm(target_authors, desc="Propagation"):
        if author in seed_domains:
            # L'auteur est un seed, on utilise son domaine directement
            final_author_domains[author] = seed_domains[author]
        elif author in reverse_adjacency:
            # L'auteur est un co-auteur, on regarde les domaines de ses voisins (seeds)
            connected_seed_domains = [seed_domains.get(s) for s in reverse_adjacency[author] if s in seed_domains]
            
            # Filtrer les domaines non-informatifs
            valid_domains = [d for d in connected_seed_domains if d not in ["Inconnu (API)", "Erreur API"]]
            
            if valid_domains:
                # Vote majoritaire
                most_common_domain = Counter(valid_domains).most_common(1)[0][0]
                final_author_domains[author] = most_common_domain
            else:
                final_author_domains[author] = "Inconnu (Propagation)"
        else:
            final_author_domains[author] = "Isolé"

    # 5. Fusionner les domaines avec le DataFrame de métriques
    df_metrics['Domaine_Dominant'] = df_metrics['Auteur'].map(final_author_domains)
    df_metrics['Domaine_Dominant'].fillna('Inconnu', inplace=True)
    
    # 6. Sauvegarder le résultat final
    df_metrics.to_csv(OUTPUT_CSV_FILE, index=False)
    
    print("\n--- ANALYSE TERMINÉE ---")
    print(f"DataFrame final sauvegardé dans '{OUTPUT_CSV_FILE}'")
    print("\nDistribution des domaines dominants :")
    print(df_metrics['Domaine_Dominant'].value_counts())

if __name__ == "__main__":
    main()
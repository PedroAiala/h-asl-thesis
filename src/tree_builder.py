import pandas as pd
import numpy as np
import random
import pickle
import graphviz
from sklearn.preprocessing import normalize
from collections import Counter

class Node:
    """Represents a node in the hierarchical binary tree."""
    def __init__(self, identities, centroid):
        self.identities = identities
        self.centroid = centroid
        self.left = None
        self.right = None
        self.is_leaf = False

    def __repr__(self):
        leaf_status = "LEAF" if self.is_leaf else "INTERNAL"
        count = len(self.identities)
        identity_str = self.identities[0] if count == 1 else f"{count} identidades"
        return f"<Node ({leaf_status}) - {identity_str}>"
    
    def inspect(self):
        """Shows details about a given node."""
           
        status = "LEAF" if self.is_leaf else "INTERNAL"
        print(f"--- Inspecionando Nó ---")
        print(f"Tipo: {status}")
        print(f"Número de Identidades: {len(self.identities)}")
        
        if self.is_leaf:
            print(f"Identidade(s): {self.identities}")
        else:
            print(f"Filho Esquerda (Left): Contém {len(self.left.identities)} identidades.")
            print(f"Filho Direita (Right): Contém {len(self.right.identities)} identidades.")


class TreeBuilder:
    """Orchestrates the construction of a hierarchical binary tree from face embeddings."""

    def __init__(self, embeddings_path, min_samples=3, max_classes= None, max_leaf_size=1):
        print("Starting TreeBuilder with strict Open-Set logic...")
        self.embeddings_path = embeddings_path
        self.min_samples = min_samples
        self.max_classes = max_classes
        self.max_leaf_size = max_leaf_size
        self.root = None
        
        self.class_centroids_map = {} 
        self.known_identities = []    
        self.background_data = []     
        
        self._load_and_prepare_data()


    def _load_and_prepare_data(self):
        """
        Loads embeddings, normalizes, applies Gunther Protocol, and computes class centroids.
        """
        print(f"   -> Loading embeddings from '{self.embeddings_path}'...")
        with open(self.embeddings_path, 'rb') as f:
            data = pickle.load(f)

        
        raw_embeddings = np.array(data['embeddings'])
        if raw_embeddings.ndim == 1: 
            raw_embeddings = np.stack(raw_embeddings)

        embeddings = normalize(raw_embeddings, norm='l2', axis=1)
        names = np.array(data['nomes'])

        
        counts = Counter(names)
        unique_names = list(counts.keys())

        known_names = [n for n in unique_names if counts[n] >= self.min_samples]
        bg_names = set([n for n in unique_names if counts[n] < self.min_samples])

        if self.max_classes and len(known_names) > self.max_classes:
            print(f"   -> Limiting known classes to {self.max_classes} as requested.")
            known_names = known_names[:self.max_classes]

        known_set = set(known_names)

        print(f"Estatísticas:")
        print(f" -> Identidades Background (<{self.min_samples} imgs): {len(bg_names)}")
        print(f" -> Identidades Conhecidas (>={self.min_samples} imgs): {len(known_set)}")

        class_vectors = {n: [] for n in known_set}
        
        for emb, name in zip(embeddings, names):
            if name in known_set:
                class_vectors[name].append(emb)
            elif name in bg_names:
                self.background_data.append(emb) 
        
        print("Calculando centróides das classes conhecidas...")
        for name in known_names:
            vecs = np.array(class_vectors[name])
            c = np.mean(vecs, axis=0)
            c = c / np.linalg.norm(c) 
            self.class_centroids_map[name] = c
            
        self.known_identities = known_names
        
        if len(self.known_identities) == 0:
            raise ValueError("Nenhuma classe conhecida encontrada com os critérios atuais.")


    def _spherical_kmeans(self, X, n_clusters=2, max_iter=100, tol=1e-4):
        X_norm = normalize(X, norm='l2', axis=1)
        if X_norm.shape[0] < n_clusters: return np.arange(X_norm.shape[0])
        
        initial_indices = np.random.choice(X_norm.shape[0], n_clusters, replace=False)
        centroids = X_norm[initial_indices]

        preds = np.zeros(X_norm.shape[0], dtype=int)
        
        for _ in range(max_iter):
            old_centroids = centroids.copy()
            similarity_matrix = np.dot(X_norm, centroids.T)
            preds = np.argmax(similarity_matrix, axis=1)
            
            new_centroids = np.zeros_like(centroids)
            for j in range(n_clusters):
                cluster_points = X_norm[preds == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else: 
                    new_centroids[j] = X_norm[np.random.choice(X_norm.shape[0])]

            centroids = normalize(new_centroids, norm='l2', axis=1)
            if np.sum((centroids - old_centroids)**2) < tol: break

        return preds


    def _build_recursive(self, current_identities, nivel):
        current_centroids = np.stack([self.class_centroids_map[name] for name in current_identities])
        
        # Centróide do nó (Média dos centróides das classes filhas)
        node_centroid = np.mean(current_centroids, axis=0)
        node_centroid = node_centroid / np.linalg.norm(node_centroid)
        
        node = Node(identities=current_identities, centroid=node_centroid)
        
        # Critério de Parada: Tamanho da folha
        if len(current_identities) <= self.max_leaf_size:
            print(f"{'  ' * nivel}L-> Nó Folha criado com {len(current_identities)} identidade(s).")
            node.is_leaf = True
            return node
            
        print(f"{'  ' * nivel}I-> Clusterizando {len(current_identities)} classes no nível {nivel}...")
        
        # Passo 2: Clusterizar os centróides 
        preds = self._spherical_kmeans(current_centroids, n_clusters=2)
        
        # Verificação de falha na divisão
        if len(np.unique(preds)) < 2:
            print(f"{'  ' * nivel}L-> Divisão não convergiu (todos no mesmo cluster). Criando nó folha.")
            node.is_leaf = True; return node
            
        identities_left = [name for i, name in enumerate(current_identities) if preds[i] == 0]
        identities_right = [name for i, name in enumerate(current_identities) if preds[i] == 1]
        
        if not identities_left or not identities_right:
            print(f"{'  ' * nivel}L-> Divisão resultou em grupo vazio. Criando nó folha.")
            node.is_leaf = True; return node
            
        node.left = self._build_recursive(identities_left, nivel + 1)
        node.right = self._build_recursive(identities_right, nivel + 1)
        return node


    def build_tree(self):
        print("\n Iniciando a construção da Árvore Hierárquica...")
        if not self.known_identities:
            raise ValueError("Nenhuma identidade válida encontrada na Galeria.")
        
        self.root = self._build_recursive(self.known_identities, nivel=0)
        print(" Árvore construída com sucesso!")


    def save_tree(self, output_path):
        print(f"\n Salvando a árvore em '{output_path}'...")
        with open(output_path, 'wb') as f:
            pickle.dump(self.root, f)
        print(f"   -> Árvore salva com sucesso em '{output_path}'.")
    

    def _add_nodes_edges(self, node, dot, parent_id=None):
        node_id = str(id(node))
        
        if node.is_leaf:
            label = f"{node.identities[0]}\n({len(node.identities)})"
            dot.node(node_id, label, shape='box', style='filled', fillcolor='lightgreen')
        else:
            label = f"{len(node.identities)} IDs"
            dot.node(node_id, label, style='filled', fillcolor='lightgray')
            
        if parent_id:
            dot.edge(parent_id, node_id)
            
        if not node.is_leaf:
            self._add_nodes_edges(node.left, dot, parent_id=node_id)
            self._add_nodes_edges(node.right, dot, parent_id=node_id)
    

    def visualize_tree_as_graph(self, output_filename='tree_visualization'):
        if not self.root:
            print("Erro: Construa a árvore antes de visualizar.")
            return
        
        dot = graphviz.Digraph(comment='Hierarchical Identity Tree', format='png')
        dot.attr(nodesep='0.5', ranksep='1')
        self._add_nodes_edges(self.root, dot)
        dot.render(output_filename, view=True)
        print(f"   -> Visualização salva em '{output_filename}.png'.")


    def find_identity(self, identity_name):
        # Função auxiliar de debug
        current = self.root
        path = ""
        while current and not current.is_leaf:
            if current.left and identity_name in set(current.left.identities):
                path += "L"
                current = current.left
            elif current.right and identity_name in set(current.right.identities):
                path += "R"
                current = current.right
            else:
                break
        print(f"Caminho para {identity_name}: {path if path else 'Não encontrado (pode ser Background)'}")
    


            


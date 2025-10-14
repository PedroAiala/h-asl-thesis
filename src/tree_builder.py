# ARQUIVO: src/tree_builder.py
# VERSÃO FINAL - DIRETA E CORRETA

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import normalize

class Node:
    """Representa um nó na árvore de identidades."""
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

class TreeBuilder:
    """Orquestra a construção da árvore binária hierárquica de identidades."""
    def __init__(self, embeddings_path, max_leaf_size=1):
        print("🚀 Inicializando TreeBuilder...")
        self.embeddings_path = embeddings_path
        self.max_leaf_size = max_leaf_size
        self.root = None
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        """
        Carrega os dados e constrói o DataFrame usando as chaves corretas.
        """
        print(f"   -> Carregando embeddings de '{self.embeddings_path}'...")
        with open(self.embeddings_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # --- CORREÇÃO DEFINITIVA ---
        # Acessa as chaves 'nomes' e 'embeddings' diretamente, sem adivinhação.
        print("   -> Construindo DataFrame a partir das chaves 'nomes' e 'embeddings'...")
        import numpy as np
        nomes = loaded_data['nomes']
        embeddings = loaded_data['embeddings']
        # Garante que cada embedding seja uma lista 1D
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        else:
            embeddings = [np.array(e).tolist() for e in embeddings]
        df = pd.DataFrame({
            'nome': nomes,
            'embedding': embeddings
        })
        # --- FIM DA CORREÇÃO ---

        print(f"   -> DataFrame criado com sucesso com {len(df)} registros.")
        print("   -> Calculando embedding médio por identidade...")
        
        # Garante que todos os embeddings são listas e têm a mesma dimensão
        df = df[df['embedding'].apply(lambda x: isinstance(x, list))].copy()
        if not df.empty:
            first_len = len(df['embedding'].iloc[0])
            df = df[df['embedding'].apply(lambda x: len(x) == first_len)].copy()
        else:
            raise ValueError("Nenhum embedding válido encontrado após o filtro.")

        mean_embeddings_series = df.groupby('nome')['embedding'].apply(
            lambda x: np.mean(np.stack(x.values), axis=0)
        )

        self.mean_embeddings_df = mean_embeddings_series.reset_index()
        self.mean_embeddings_df.rename(columns={'embedding': 'mean_embedding'}, inplace=True)
        
        self.embeddings_map = {row['nome']: row['mean_embedding'] for _, row in self.mean_embeddings_df.iterrows()}
        self.all_identities = self.mean_embeddings_df['nome'].tolist()
        
        print(f"   -> Dados preparados. {len(self.all_identities)} identidades únicas encontradas.")

    def _spherical_kmeans(self, X, n_clusters=2, max_iter=100, tol=1e-4):
        X_norm = normalize(X, norm='l2', axis=1)
        if X_norm.shape[0] < n_clusters: return np.arange(X_norm.shape[0])
        
        initial_indices = np.random.choice(X_norm.shape[0], n_clusters, replace=False)
        centroids = X_norm[initial_indices]
        
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
        current_embeddings = np.stack([self.embeddings_map[name] for name in current_identities])
        current_centroid = np.mean(current_embeddings, axis=0)
        node = Node(identities=current_identities, centroid=current_centroid)
        
        if len(current_identities) <= self.max_leaf_size:
            print(f"{'  ' * nivel}L-> Nó Folha criado com a identidade: {current_identities[0]}")
            node.is_leaf = True
            return node
            
        print(f"{'  ' * nivel}I-> Dividindo nó com {len(current_identities)} identidades no nível {nivel}...")
        preds = self._spherical_kmeans(current_embeddings)
        
        if len(np.unique(preds)) < 2:
            print(f"{'  ' * nivel}L-> Divisão não foi possível. Criando nó folha.")
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
        print("\n🌳 Iniciando a construção da Árvore Hierárquica...")
        if not self.all_identities:
            raise ValueError("Nenhuma identidade válida encontrada para construir a árvore.")
        self.root = self._build_recursive(self.all_identities, nivel=0)
        print("✅ Árvore construída com sucesso!")

    def save_tree(self, output_path):
        print(f"\n💾 Salvando a árvore em '{output_path}'...")
        with open(output_path, 'wb') as f:
            pickle.dump(self.root, f)
        print(f"   -> Árvore salva com sucesso em '{output_path}'.")
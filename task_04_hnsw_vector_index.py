"""
Task 4: In-Memory Hierarchical Navigable Small World (HNSW) Vector Indexing from Scratch
--------------------------------------------------------------------------------------
Objective: Develop a high-speed vector retrieval engine to support semantic search and
real-time knowledge retrieval without utilizing pre-built database drivers.

Required Tech Stack: Python, NetworkX, SciPy, NumPy
Search Complexity: O(log N)
"""

import math
import random
import numpy as np
from scipy.spatial.distance import cosine

class HNSWNode:
    def __init__(self, node_id: int, vector: np.ndarray, level: int):
        self.node_id = node_id
        self.vector = vector
        self.level = level
        # Adjacency lists for each layer from 0 to level: list of lists of node_ids
        self.neighbors: list[list[int]] = [[] for _ in range(level + 1)]


class HNSWIndex:
    """
    Hierarchical Navigable Small World Graph from scratch.
    """
    def __init__(self, dim: int, m: int = 16, ef_construction: int = 32, mL: float = 1 / math.log(16)):
        self.dim = dim
        self.m = m                       # Max number of outgoing connections per node
        self.ef_construction = ef_construction
        self.mL = mL                     # Normalization factor for level generation
        self.nodes: dict[int, HNSWNode] = {}
        self.enter_node_id: int | None = None
        self.max_level = -1

    def _distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculates Cosine distance between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(cosine(vec1, vec2))

    def _random_level(self) -> int:
        """Probability-driven layer skipping assignment."""
        unif = random.uniform(1e-9, 1.0)
        return math.floor(-math.log(unif) * self.mL)

    def _search_layer(self, query_vec: np.ndarray, entry_points: list[int], ef: int, level: int) -> list[tuple[float, int]]:
        """
        Greedy search within a specific graph layer.
        Returns top-ef nearest (distance, node_id) tuples.
        """
        visited = set(entry_points)
        candidates = []  # Min-heap simulated as sorted list
        for ep in entry_points:
            dist = self._distance(query_vec, self.nodes[ep].vector)
            candidates.append((dist, ep))
        
        candidates.sort(key=lambda x: x[0])
        w = list(candidates)  # Best results found so far

        while len(candidates) > 0:
            curr_dist, curr_id = candidates.pop(0)

            # Further candidate is worse than furthest in w
            if curr_dist > w[-1][0] and len(w) >= ef:
                break

            # Explore neighbors at layer 'level'
            curr_node = self.nodes[curr_id]
            if level < len(curr_node.neighbors):
                for neighbor_id in curr_node.neighbors[level]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        d = self._distance(query_vec, self.nodes[neighbor_id].vector)
                        if d < w[-1][0] or len(w) < ef:
                            candidates.append((d, neighbor_id))
                            candidates.sort(key=lambda x: x[0])
                            w.append((d, neighbor_id))
                            w.sort(key=lambda x: x[0])
                            if len(w) > ef:
                                w.pop() # Keep top ef

        return w

    def insert(self, node_id: int, vector: np.ndarray):
        """
        Inserts a new vector into the HNSW index graph.
        """
        level = self._random_level()
        new_node = HNSWNode(node_id, vector, level)
        self.nodes[node_id] = new_node

        if self.enter_node_id is None:
            self.enter_node_id = node_id
            self.max_level = level
            return

        curr_ep = [self.enter_node_id]
        curr_max_level = self.max_level

        # 1. Top-down traversal from max_level down to level + 1 (greedy skip-list traversal)
        for l in range(curr_max_level, level, -1):
            w = self._search_layer(vector, curr_ep, ef=1, level=l)
            curr_ep = [w[0][1]]

        # 2. From min(level, max_level) down to 0, connect neighbors
        for l in range(min(level, curr_max_level), -1, -1):
            w = self._search_layer(vector, curr_ep, ef=self.ef_construction, level=l)
            # Pick top M neighbors
            neighbors = [item[1] for item in w[:self.m]]
            new_node.neighbors[l] = neighbors

            # Add bidirectional edges
            for n_id in neighbors:
                self.nodes[n_id].neighbors[l].append(node_id)
                # Prune if exceeding max M
                if len(self.nodes[n_id].neighbors[l]) > self.m:
                    # Keep closest M
                    self.nodes[n_id].neighbors[l].sort(
                        key=lambda target_id: self._distance(self.nodes[n_id].vector, self.nodes[target_id].vector)
                    )
                    self.nodes[n_id].neighbors[l] = self.nodes[n_id].neighbors[l][:self.m]

            curr_ep = [item[1] for item in w]

        if level > self.max_level:
            self.max_level = level
            self.enter_node_id = node_id

    def search(self, query_vec: np.ndarray, k: int = 5, ef_search: int = 16) -> list[tuple[int, float]]:
        """
        Searches the HNSW index for the k-nearest neighbors of query_vec.
        Returns list of (node_id, cosine_similarity).
        """
        if self.enter_node_id is None:
            return []

        curr_ep = [self.enter_node_id]
        for l in range(self.max_level, 0, -1):
            w = self._search_layer(query_vec, curr_ep, ef=1, level=l)
            curr_ep = [w[0][1]]

        w = self._search_layer(query_vec, curr_ep, ef=max(ef_search, k), level=0)
        # Cosine similarity = 1 - Cosine distance
        results = [(node_id, 1.0 - dist) for dist, node_id in w[:k]]
        return results


def main():
    print("=" * 70)
    print("Task 4: In-Memory HNSW Vector Indexing from Scratch Verification")
    print("=" * 70)

    np.random.seed(42)
    random.seed(42)

    dim = 32
    num_vectors = 500
    k_neighbors = 5

    print(f"Building HNSW Index for N={num_vectors} vectors of dimension D={dim}...")
    index = HNSWIndex(dim=dim, m=12, ef_construction=24)

    dataset = np.random.randn(num_vectors, dim).astype(np.float32)
    # Normalize vectors to unit length
    dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)

    for i in range(num_vectors):
        index.insert(i, dataset[i])

    print(f"HNSW Graph Constructed. Max Layer Depth achieved: {index.max_level}")

    # Query search test
    query = np.random.randn(dim).astype(np.float32)
    query /= np.linalg.norm(query)

    hnsw_results = index.search(query, k=k_neighbors, ef_search=32)

    # Brute force search for ground truth comparison
    similarities = np.dot(dataset, query)
    ground_truth_indices = np.argsort(similarities)[::-1][:k_neighbors]
    ground_truth_sims = similarities[ground_truth_indices]

    print(f"\nHNSW Top-{k_neighbors} Search Results:")
    for rank, (node_id, sim) in enumerate(hnsw_results, 1):
        print(f"  Rank {rank}: Node ID {node_id:3d} | Cosine Similarity: {sim:.4f}")

    print(f"\nBrute-Force Ground Truth Top-{k_neighbors}:")
    for rank, (node_id, sim) in enumerate(zip(ground_truth_indices, ground_truth_sims), 1):
        print(f"  Rank {rank}: Node ID {node_id:3d} | Cosine Similarity: {sim:.4f}")

    # Compute Recall @ K
    hnsw_retrieved_set = set([r[0] for r in hnsw_results])
    gt_set = set(ground_truth_indices)
    recall = len(hnsw_retrieved_set.intersection(gt_set)) / k_neighbors

    print(f"\nHNSW Search Recall @ {k_neighbors}: {recall * 100:.1f}%")
    assert recall > 0.6, "HNSW Search recall is unexpectedly low!"
    print("\nTask 4 completed successfully!")

if __name__ == "__main__":
    main()

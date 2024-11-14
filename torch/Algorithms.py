import pandas as pd
import numpy as np
from anytree import Node, RenderTree, LevelOrderIter


class TreeInitialization:
    def __init__(self, tree):
        self.tree = tree
        self.nodes = list(LevelOrderIter(tree))
        for i, node in enumerate(self.nodes):
            node.id = i
        self.df = self.create_tree_df()

    def create_tree_df(self):
        df = pd.DataFrame({
            "node": [node.name for node in self.nodes],
            "id": [node.id for node in self.nodes],
            "level": [node.depth for node in self.nodes],
            "isLeaf": [node.is_leaf for node in self.nodes],
            "parent": [node.parent.id if node.parent else -1 for node in self.nodes]
        })
        return df



class TreeEmbedding:
    def __init__(self, tree):
        self.tree_init = TreeInitialization(tree)
        self.tree = self.tree_init.tree
        self.nodes = self.tree_init.nodes
        self.df = self.tree_init.df
        self.xi = self.hier_emb()


    def multi_emb0(self, c, q):
        """Algorithm 1: Label embedding in multicategory classification"""
        if q < 2:
            raise ValueError("The number of leaves must be at least 2.")

        Xi = np.zeros((q-1, q))
        Xi[0, 0] = -c/2; Xi[0, 1] = c/2
        
        for m in range(2, q):
            d_in = np.mean(Xi[:(m-1), :m], axis=1) - Xi[:(m-1), m-1]
            d = np.sqrt(np.sum(d_in**2))
            a = np.sqrt(c**2 - d**2)
            e = np.zeros(m); e[-1] = 1
            Xi[:m, m] = np.mean(Xi[:m, :m], axis=1) + a * e

        Xi -= Xi.mean(axis=1, keepdims=True)

        return Xi


    def multi_emb(self, t, q):
        """Algorithm 1: Label embedding in multicategory classification w/ scaling"""
        cq = t * np.sqrt((2 * q) / (q - 1))
        return self.multi_emb0(cq, q)


    def hier_emb(self, t=1, delta=np.sqrt(5)):
        """Algorithm 2: Label embedding in hierarchical classification"""
        n_nodes = self.tree.size
        n_leafs = len(self.tree.leaves)

        Xi = np.zeros((n_leafs-1, n_nodes-1))
        start_row = start_col = 0
        for node in self.nodes:
            n_child = len(node.children)
            if n_child > 0:
                emb = self.multi_emb(t/ delta**node.depth, n_child)
                end_row = start_row + n_child - 1
                end_col = start_col + n_child
                Xi[start_row:end_row, start_col:end_col] = emb
                start_row, start_col = end_row, end_col

        for i, node in enumerate(self.nodes[1:]):
            if node.parent.id > 0:
                Xi[:, i] += Xi[:, node.parent.id-1]

        return Xi
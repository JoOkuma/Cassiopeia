"""
This file stores a subclass of CassiopeiaSolver, the GreedySolver. This class
represents the structure of top-down algorithms that build the reconstructed 
tree by recursively splitting the set of samples based on some split criterion.
"""
import logging

import abc
import networkx as nx
import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Union

from cassiopeia.solver import CassiopeiaSolver
from cassiopeia.solver import solver_utilities


class GreedySolver(CassiopeiaSolver.CassiopeiaSolver):
    """
    GreedySolver is an abstract class representing the structure of top-down
    inference algorithms. The solver procedure contains logic to build a tree
    from the root by recursively partitioning the set of samples. Each subclass
    will implement "perform_split", which is the procedure for successively
    partioning the sample set.

    Args:
        character_matrix: A character matrix of observed character states for
            all samples
        missing_char: The character representing missing values
        meta_data: Any meta data associated with the samples
        priors: Prior probabilities of observing a transition from 0 to any
            state for each character
        prior_transformation: A function defining a transformation on the priors
            in forming weights

    Attributes:
        character_matrix: The character matrix describing the samples
        missing_char: The character representing missing values
        meta_data: Data table storing meta data for each sample
        priors: Prior probabilities of character state transitions
        weights: Weights on character/mutation pairs, derived from priors
        tree: The tree built by `self.solve()`. None if `solve` has not been
            called yet
        unique_character_matrix: A character matrix with duplicate rows filtered
            out, converted to a numpy array for efficient indexing
        index_to_name: A dictionary mapping sample names to their integer
            indices in the original character matrix, for efficient indexing
        name_to_index: A dictionary mapping integer indices of samples in
            the original character matrix to their names
        duplicate_groups: A mapping of samples to the set of duplicates that
            share the same character vector. Uses the original sample names
    """

    def __init__(
        self,
        character_matrix: pd.DataFrame,
        missing_char: int,
        meta_data: Optional[pd.DataFrame] = None,
        priors: Optional[Dict[int, Dict[int, float]]] = None,
        prior_transformation: str = "negative_log",
    ):

        super().__init__(character_matrix, missing_char, meta_data, priors)
        if priors:
            self.weights = solver_utilities.transform_priors(
                priors, prior_transformation
            )
        else:
            self.weights = None

        unique_character_matrix = self.character_matrix.drop_duplicates()
        self.unique_character_matrix = unique_character_matrix.to_numpy()

        self.index_to_name = dict(
            zip(
                range(unique_character_matrix.shape[0]),
                unique_character_matrix.index,
            )
        )
        self.name_to_index = dict(
            zip(
                unique_character_matrix.index,
                range(unique_character_matrix.shape[0]),
            )
        )

        self.duplicate_groups = (
            character_matrix[character_matrix.duplicated(keep=False) == True]
            .reset_index()
            .groupby(character_matrix.columns.tolist())["index"]
            .agg(["first", tuple])
            .set_index("first")["tuple"]
            .to_dict()
        )

    def perform_split(
        self,
        samples: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Performs a partition of the samples.

        Args:
            samples: A list of samples, represented by their string names

        Returns:
            A tuple of lists, representing the left and right partition groups
        """
        pass

    def solve(self) -> nx.DiGraph:
        """Implements a top-down greedy solving procedure.

        The procedure recursively splits a set of samples to build a tree. At
        each partition of the samples, an ancestral node is created and each
        side of the partition is placed as a daughter clade of that node. This
        continues until each side of the partition is comprised only of single
        samples. If an algorithm cannot produce a split on a set of samples,
        then those samples are placed as sister nodes and the procedure
        terminates, generating a polytomy in the tree.

        Returns:
            A networkx directed graph representing the reconstructed tree
        """

        # A helper function that builds the subtree given a set of samples
        def _solve(samples):
            if len(samples) == 1:
                return samples[0]
            # Finds the best partition of the set given the split criteria
            clades = list(self.perform_split(samples))
            # Generates a root for this subtree with a unique int identifier
            root = (
                len(self.tree.nodes)
                - len(self.unique_character_matrix)
                + self.character_matrix.shape[0]
            )
            self.tree.add_node(root)

            for clade in clades:
                if len(clade) == 0:
                    clades.remove(clade)

            # If unable to return a split, generate a polytomy and return
            if len(clades) == 1:
                for clade in clades[0]:
                    self.tree.add_edge(root, clade)
                return root
            # Recursively generate the subtrees for each daughter clade
            for clade in clades:
                child = _solve(clade)
                self.tree.add_edge(root, child)
            return root

        self.tree = nx.DiGraph()
        samples = list(self.name_to_index.keys())
        for i in samples:
            self.tree.add_node(i)
        _solve(samples)
        # Collapse 0-mutation edges and append duplicate samples
        self.tree = solver_utilities.collapse_tree(
            self.tree, True, self.character_matrix, self.missing_char
        )
        self.tree = self.add_duplicates_to_tree(self.tree)
        return self.tree

    def compute_mutation_frequencies(
        self, samples: List[int]
    ) -> Dict[int, Dict[int, int]]:
        """Computes the number of samples in a character matrix that have each
        character/state mutation.

        Computes the frequency of each mutation in the character data of the
        sample set. Duplicate cells have their frequencies included. Generates
        a dictionary that maps each character to a dictionary of state/sample
        frequency pairs, allowing quick lookup.

        Args:
            samples: The set of relevant samples in calculating frequencies,
                represented by their string names

        Returns:
            A dictionary containing frequency information for each character/state
            pair

        """
        subset_cm = self.unique_character_matrix[samples, :]
        freq_dict = {}
        for char in range(subset_cm.shape[1]):
            char_dict = {}
            state_counts = np.unique(subset_cm[:, char], return_counts=True)
            for i in range(len(state_counts[0])):
                state = state_counts[0][i]
                count = state_counts[1][i]
                char_dict[state] = count
            if self.missing_char not in char_dict:
                char_dict[self.missing_char] = 0
            freq_dict[char] = char_dict

        return freq_dict

    def add_duplicates_to_tree(self, tree: nx.DiGraph) -> nx.DiGraph:
        """Takes duplicate samples and places them in the tree.

        Places samples removed in removing duplicates in the tree as sisters
        to the corresponding cells that share the same mutations.

        Args:
            tree: The tree to have duplicates added to
        Returns:
            A tree with duplicates added
        """
        for i in self.duplicate_groups:
            new_internal_node = (
                max([i for i in tree.nodes if type(i) == int]) + 1
            )
            nx.relabel_nodes(tree, {i: new_internal_node}, copy=False)
            for duplicate in self.duplicate_groups[i]:
                tree.add_edge(new_internal_node, duplicate)

        return tree

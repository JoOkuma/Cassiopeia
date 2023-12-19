import logging
import heapq
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from cassiopeia.data.CassiopeiaTree import CassiopeiaTree
from cassiopeia.solver.CassiopeiaSolver import CassiopeiaSolver
from cassiopeia.mixins import logger


try:
    import cupy as xp
except ImportError:
    xp = np


def _lazy_M_constraints(x: gp.tupledict, B: int, C: int) -> Callable:

    def _callback(m: gp.Model, where: int) -> None:
        # if where == GRB.Callback.MIPSOL:
        if where != GRB.Callback.MIPSOL and where != GRB.Callback.MIPNODE:
            return

        elif where == GRB.Callback.MIPSOL:
            sols = m.cbGetSolution(x.values())
            add_constr_func = m.cbLazy

        elif where == GRB.Callback.MIPNODE:
            if m.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return

            sols = m.cbGetNodeRel(x.values())
            add_constr_func = m.cbCut

        sols = np.reshape(sols, (B, C))
        sols = xp.asarray(sols)

        C1 = np.argmax(sols[:, None, ...] - sols[None, :], axis=-1)
        C2 = np.argmax(sols[:, None, ...] + sols[None, :], axis=-1)
        C3 = np.argmax(-sols[:, None, ...] + sols[None, :], axis=-1)

        if hasattr(C1, "get"):
            C1 = C1.get()
            C2 = C2.get()
            C3 = C3.get()

        for b1 in range(B):
            for b2 in range(b1):
                c1 = C1[b1, b2]
                c2 = C2[b1, b2]
                c3 = C3[b1, b2]
                add_constr_func(
                    (x[b1, c1] - x[b2, c1]) + (x[b1, c2] + x[b2, c2]) + (x[b2, c3] - x[b1, c3])  <= 3
                )

    return _callback


def lazy_super_tree_ILP(
    character_matrix: np.ndarray,
    cell_weights: np.ndarray,
    deletion_weight: Optional[float],
) -> np.ndarray:
    """
    Fill in missing reads in the barcode-leaf table using formulation 4.1 from [1]_.

    M constraints

       B1      B2
      /  \    /  \
     /    \  /    \
    C1     C2      C3

    x_{b1c1} + x_{b1c2} + x_{b2c2} + x_{b2c3} <= 3 + x_{b2c1} + x_{b1c3}

    or

    x_{b1c1} + x_{b1c2} - x_{b1c3} + x_{b2c2} + x_{b2c3} - x_{b2c1} <= 3

    [1]_ https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=8d9ee958d45bb46243e3c43ffd8c0b5fad658747

    Parameters
    ----------
    character_matrix : np.ndarray
        Character matrix with shape (cells, barcode)
    
    cell_weights : np.ndarray
        Cell weights with shape (cells,)
    
    deletion_weight : float
        Deletion weight used to balance insertion vs deletion.
        If None, deletion is not allowed and not penalized.

    Returns
    -------
    np.ndarray
        The filled-in table.
    """
    assert character_matrix.dtype == bool

    # initial implementation was for a character matrix with shape (barcode, cells)
    tb = character_matrix.T

    B = tb.shape[0]
    C = tb.shape[1]

    m = gp.Model()
    m.Params.LazyConstraints = 1

    x = m.addVars(B, C, vtype=GRB.BINARY)

    if deletion_weight is None:
        # setting existing barcodes reads as hard constraints
        m.addConstrs(
            x[b, c] == 1
            for b in range(B)
            for c in range(C)
            if tb[b, c]
        )
        # objective without deletion weight
        m.setObjective(
            gp.quicksum(
                x[b, c] * cell_weights[c]
                for b in range(B)
                for c in range(C)
            )
        )

    else:
        insertion_weight = 1 - deletion_weight

        m.setObjective(
            gp.quicksum(
                x[b, c] * cell_weights[c] * (-deletion_weight if tb[b, c] else insertion_weight)
                for b in range(B)
                for c in range(C)
            )
        )

    lazy_constraints = _lazy_M_constraints(x, B, C)
    m.optimize(lazy_constraints)

    x_hat = m.getAttr("X", x)

    tb_hat = np.zeros_like(tb)
    for b in range(B):
        for c in range(C):
            tb_hat[b, c] = x_hat[b, c] > 0.5

    return tb_hat.T


def character_matrix_to_tree(
    character_matrix: pd.DataFrame,
    character_size: np.ndarray,
    duplicated_cells: list[list[str]],
) -> nx.DiGraph:

    G = nx.DiGraph()

    height = character_size.T @ character_matrix
    height = np.ceil(np.log2(height)).astype(int)

    # construct candidate graph
    seen = set()

    for i, bc in enumerate(character_matrix.columns):
        G.add_node(bc, height=height[i])

    for cell in character_matrix.index:
        G.add_node(cell, height=0)

        for i, bc in enumerate(character_matrix.columns):
            if character_matrix.loc[cell, bc]:
                G.add_edge(bc, cell, height=height[i])

                for j, other_bc in enumerate(character_matrix.columns):
                    if height[i] < height[j] and (other_bc, bc) not in seen and character_matrix.loc[cell, other_bc]:
                        G.add_edge(other_bc, bc, weight=height[j] - height[i])
                        seen.add((other_bc, bc))
    
    nodes_without_parent = [node for node, in_degree in G.in_degree() if in_degree == 0]

    if len(nodes_without_parent) > 1:
        root = "roots"
        G.add_node(root, height=height.max() + 1)
        for node in nodes_without_parent:
            G.add_edge(root, node, weight=0)
    else:
        root = nodes_without_parent[0]
    
    # compute tree
    T = nx.DiGraph()
    queue = []
    for cell in character_matrix.index:
        T.add_node(cell, height=0)
        queue.append((0, cell))

    seen = set()

    while queue:
        _, node = heapq.heappop(queue)

        if node in seen or node == root:
            continue

        seen.add(node)

        parent = sorted(G.predecessors(node), key=lambda x: G.nodes[x]["height"])[0]

        if node not in T.nodes:
            T.add_node(node, height=G.nodes[node]["height"])

        if parent not in T.nodes:
            T.add_node(parent, height=G.nodes[parent]["height"])

        T.add_edge(parent, node)

        heapq.heappush(queue, (G.nodes[parent]["height"], parent))

    for c, siblings in zip(character_matrix.index, duplicated_cells):
        if len(siblings) == 1:
            continue

        # deleting bundled sibling node
        parent = next(T.predecessors(c))
        parent_height = T.nodes[parent]["height"]
        T.remove_edge(parent, c)

        # adding new bundled sibling node
        new_artificial_node = c + "_dup"
        height = np.ceil(np.log2(len(siblings)))
        height = min(height, parent_height - 1e-6)  # hack to avoid height equal to parent

        T.add_node(new_artificial_node, height=int(height))
        T.add_edge(parent, new_artificial_node)

        for sibling in siblings:
            T.add_node(sibling, height=0)
            T.add_edge(new_artificial_node, sibling)

    for node in T.nodes:
        T.nodes[node]["height"] = -int(T.nodes[node]["height"])

    return T


class SuperTreeSolver(CassiopeiaSolver):
    def __init__(self):
        super().__init__()

    @logger.namespaced("SuperTreeSolver")
    def solve(
        self,
        cassiopeia_tree: CassiopeiaTree,
        deletion_weight: Optional[float] = 0.999,
        layer: Optional[str] = None,
        collapse_mutationless_edges: bool = True,
        logfile: str = "stdout.log",
    ):
        """
        Solves the inference problem.

        Args:
            cassiopeia_tree: CassiopeiaTree storing character information for
                phylogenetic inference.
            deletion_weight: Weight to assign to deletion events. If None,
                deletion is not allowed and not penalized.
            layer: Layer storing the character matrix for solving. If None, the
                default character matrix is used in the CassiopeiaTree.
            collapse_mutationless_edges: Must be True, kept for compatibility.
            logfile: File location to log output.
        """
        pass
        if not collapse_mutationless_edges:
            raise NotImplementedError(
                "SuperTreeSolver currently only supports collapsing mutationless edges."
            )

        handler = logging.FileHandler(logfile)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        # cell x barcode matrix
        if layer:
            char_matrix = cassiopeia_tree.layers[layer]
        else:
            char_matrix = cassiopeia_tree.character_matrix

        logger.info("Character matrix has shape: %s", char_matrix.shape)

        indices = []
        unique_cells = []
        for _, g in char_matrix.groupby(char_matrix.columns.tolist(), as_index=False):
            indices.append(g.index)
            unique_cells.append(g.iloc[0])

        weights = np.asarray([len(i) for i in indices])
        unique_char_matrix = pd.DataFrame(unique_cells)
        
        logger.info("Unique character matrix has shape: %s", unique_char_matrix.shape)

        reconstructed_unique_char_matrix = lazy_super_tree_ILP(
            unique_char_matrix.to_numpy(dtype=bool), weights, deletion_weight,
        )

        reconstructed_unique_char_matrix = pd.DataFrame(
            reconstructed_unique_char_matrix, columns=unique_char_matrix.columns, index=unique_char_matrix.index
        )

        # character_matrix_to_tree(reconstructed_unique_char_matrix, weights, indices)
        cassiopeia_tree.populate_tree(
          character_matrix_to_tree(reconstructed_unique_char_matrix, weights, indices)
        )

        cassiopeia_tree.collapse_unifurcations()

        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        # # Plot first matrix
        # img1 = axs[0].imshow(unique_char_matrix.to_numpy(float), cmap='magma')
        # axs[0].set_title("Original")
        # fig.colorbar(img1, ax=axs[0])

        # # Plot second matrix
        # img2 = axs[1].imshow(reconstructed_unique_char_matrix.to_numpy(float), cmap='magma')
        # axs[1].set_title("Reconstructed")
        # fig.colorbar(img2, ax=axs[1])

        # plt.show()

        logger.removeHandler(handler)

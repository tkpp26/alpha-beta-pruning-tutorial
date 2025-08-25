# https://github.com/aimacode/aima-python/blob/master/games4e.py
# https://networkx.org/documentation/stable/reference/classes/digraph.html

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import math, time, random
from contextlib import contextmanager

# =========================
# Data structures & logging
# =========================
@dataclass
class Node:
    id: str
    kind: str                    # "MAX", "MIN", "LEAF"
    value: Optional[int]
    children: List[str] = field(default_factory=list)
    # provenance so each node maps to a board move
    move: Optional[int] = None   # 0..8 index that produced THIS node (None for ROOT)
    by: Optional[str] = None     # "X"/"O" who played that move

def default_tree() -> Dict[str, Node]:
    # Small toy tree for the tutorial tab
    L1 = Node("L1", "LEAF", 3)
    L2 = Node("L2", "LEAF", 12)
    L3 = Node("L3", "LEAF", 8)
    M1 = Node("M1", "LEAF", 2)
    M2 = Node("M2", "LEAF", 4)
    M3 = Node("M3", "LEAF", 6)
    R1 = Node("R1", "LEAF", 14)
    R2 = Node("R2", "LEAF", 5)
    R3 = Node("R3", "LEAF", 2)
    A = Node("A", "MIN", None, ["L1","L2","L3"])
    B = Node("B", "MIN", None, ["M1","M2","M3"])
    C = Node("C", "MIN", None, ["R1","R2","R3"])
    ROOT = Node("ROOT", "MAX", None, ["A","B","C"])
    return {n.id: n for n in [ROOT, A, B, C, L1, L2, L3, M1, M2, M3, R1, R2, R3]}

@dataclass
class Step:
    description: str
    current: str
    discovered_values: Dict[str, int]
    frontier_eval: Dict[str, int]
    alpha_beta: Dict[str, Tuple[Optional[int], Optional[int]]]
    pruned_edges: List[Tuple[str, str]]
    back_edge: Optional[Tuple[str, str]] = None
    back_value: Optional[int] = None

# =========================
# Minimax / Alpha–Beta logs
# =========================
def minimax_steps(nodes: Dict[str, Node]) -> List[Step]:
    steps: List[Step] = []
    discovered: Dict[str, int] = {}
    frontier: Dict[str, int] = {}
    ab: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    pruned: List[Tuple[str, str]] = []

    def log(desc, current, back_edge=None, back_value=None):
        steps.append(Step(
            desc, current, discovered.copy(), frontier.copy(),
            ab.copy(), pruned.copy(), back_edge=back_edge, back_value=back_value
        ))

    def dfs(u: str) -> int:
        node = nodes[u]
        if node.kind == "LEAF":
            frontier[u] = node.value
            log(f"Evaluate leaf {u} = {node.value}.", u)
            return node.value
        vals = []
        log(f"Enter {node.kind} node {u}.", u)
        for ch in node.children:
            val = dfs(ch)
            vals.append(val)
            best = max(vals) if node.kind == "MAX" else min(vals)
            discovered[u] = best
            log(f"Back up {best} to {u} after exploring {ch}.", u,
                back_edge=(ch, u), back_value=best)
        return discovered[u]

    dfs("ROOT")
    log("Minimax complete.", "ROOT")
    return steps

def alphabeta_steps(nodes: Dict[str, Node]) -> List[Step]:
    steps: List[Step] = []
    discovered: Dict[str, int] = {}
    frontier: Dict[str, int] = {}
    ab: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    pruned: List[Tuple[str, str]] = []

    def log(desc, current, back_edge=None, back_value=None):
        steps.append(Step(
            desc, current, discovered.copy(), frontier.copy(),
            ab.copy(), pruned.copy(), back_edge=back_edge, back_value=back_value
        ))

    def dfs(u: str, alpha: int, beta: int) -> int:
        node = nodes[u]
        ab[u] = (alpha, beta)
        if node.kind == "LEAF":
            frontier[u] = node.value
            log(f"Evaluate leaf {u} = {node.value}.", u)
            return node.value

        if node.kind == "MAX":
            value = float("-inf")
            log(f"Enter MAX {u} with (α={alpha}, β={beta}).", u)
            for ch in node.children:
                val = dfs(ch, alpha, beta)
                value = max(value, val)
                discovered[u] = value
                alpha = max(alpha, value)
                ab[u] = (alpha, beta)
                log(f"Update {u}: value={value}, α={alpha}, β={beta} after {ch}.", u,
                    back_edge=(ch, u), back_value=value)
                if alpha >= beta:
                    idx = node.children.index(ch)
                    for pruned_child in node.children[idx+1:]:
                        pruned.append((u, pruned_child))
                        log(f"Prune edge {u}→{pruned_child} (α ≥ β).", u)
                    break
            return value
        else:
            value = float("inf")
            log(f"Enter MIN {u} with (α={alpha}, β={beta}).", u)
            for ch in node.children:
                val = dfs(ch, alpha, beta)
                value = min(value, val)
                discovered[u] = value
                beta = min(beta, value)
                ab[u] = (alpha, beta)
                log(f"Update {u}: value={value}, α={alpha}, β={beta} after {ch}.", u,
                    back_edge=(ch, u), back_value=value)
                if alpha >= beta:
                    idx = node.children.index(ch)
                    for pruned_child in node.children[idx+1:]:
                        pruned.append((u, pruned_child))
                        log(f"Prune edge {u}→{pruned_child} (α ≥ β).", u)
                    break
            return value

    dfs("ROOT", float("-inf"), float("inf"))
    log("Alpha–Beta complete.", "ROOT")
    return steps

# =========================
# Diagram helpers
# =========================
def _is_trivial_tree(nodes: Dict[str, Node]) -> bool:
    """True if there is only a root leaf or there are no edges to draw."""
    if not nodes:
        return True
    root = nodes.get("ROOT")
    if root is None:
        return False
    return (root.kind == "LEAF") and (not root.children)

def hierarchy_pos(G, root, width=2.8, vert_gap=0.28, vert_loc=1.0, xcenter=0.0, sibling_sep=0.0):
    """Place nodes in a tidy top-down hierarchy."""
    from collections import defaultdict
    children = defaultdict(list)
    parent = {}
    for u, v in G.edges():
        children[u].append(v)
        parent[v] = u

    def is_leaf(n): return len(children[n]) == 0

    def count_leaves(n):
        if is_leaf(n): return 1
        return sum(count_leaves(c) for c in children[n])

    subtree_leaves = {}
    def dfs(n):
        subtree_leaves[n] = count_leaves(n)
        for c in children[n]:
            dfs(c)
    dfs(root)

    pos = {}
    def place(n, left, right, y):
        x = (left + right) / 2.0
        pos[n] = (x, y)
        k = len(children[n])
        if k == 0:
            return
        total = right - left
        total_gaps = sibling_sep * (k - 1)
        avail = max(total - total_gaps, 0.0)
        start = left
        for i, c in enumerate(children[n]):
            frac = subtree_leaves[c] / subtree_leaves[n]
            w = avail * frac
            place(c, start, start + w, y - vert_gap)
            start += w
            if i < k - 1:
                start += sibling_sep

    place(root, xcenter - width/2, xcenter + width/2, vert_loc)
    return pos

def _fmt_ab(a, b):
    def f(x):
        if x == float("-inf"): return "-∞"
        if x == float("inf"):  return "∞"
        return str(x)
    return f(a), f(b)

def draw_tree(nodes: Dict[str, Node], step: Step, show_alpha_beta: bool, title: str, compact: bool = True):
    # sizing
    n_leaves = sum(1 for n in nodes.values() if n.kind == "LEAF")
    fig_w = 8.0 if compact else 10.5
    fig_h = 5.0 if compact else 6.2
    node_size = max(600, 1600 - 40 * n_leaves) if compact else 2000
    font_size = 8 if compact else 10
    margin = 0.05 if compact else 0.15

    G = nx.DiGraph()
    for n in nodes.values():
        for ch in n.children:
            G.add_edge(n.id, ch)

    pos = hierarchy_pos(G, "ROOT", width=5.0, vert_gap=0.34, sibling_sep=0.10)

    pruned_set = set(step.pruned_edges)
    pruned_targets = {v for (_, v) in pruned_set}

    colors_map, labels = {}, {}
    for nid, n in nodes.items():
        if nid == step.current:
            colors_map[nid] = "#ffd166"  # current
        elif nid in pruned_targets:
            colors_map[nid] = "#eeeeee"  # pruned child
        elif (nid in step.discovered_values) or (nid in step.frontier_eval):
            colors_map[nid] = "#90ee90"  # visited/evaluated
        else:
            colors_map[nid] = "#c5d6ff"  # untouched

        # label
        if n.kind == "LEAF":
            base = f"{nid}\n{n.value}"
        else:
            base = f"{nid} ({n.kind})"
            if nid in step.discovered_values:
                base += f"\n= {step.discovered_values[nid]}"

        # move provenance (1..9 cell shown)
        if n.move is not None and n.by is not None:
            base += f"\n m={n.move+1} by {n.by}"

        if show_alpha_beta and nid in step.alpha_beta:
            a, b = step.alpha_beta[nid]
            a_str, b_str = _fmt_ab(a, b)
            base += f"\nα={a_str}, β={b_str}"
        labels[nid] = base

    edge_colors = ["#cccccc" if (u, v) in pruned_set else "#333333" for u, v in G.edges()]

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()
    ax.margins(margin)

    nx.draw(G, pos, with_labels=False,
            node_color=[colors_map[n] for n in G.nodes()],
            edge_color=edge_colors, node_size=node_size, arrows=False)

    nx.draw_networkx_labels(
        G, pos, labels, font_size=font_size,
        bbox=dict(facecolor="none", edgecolor="none", alpha=0.7, pad=0.3 if compact else 0.5)
    )

    if step.back_edge:
        child, parent = step.back_edge
        (x0, y0) = pos[child]; (x1, y1) = pos[parent]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", lw=1.6 if compact else 2, color="#d55",
                            linestyle=(0, (6, 4)), connectionstyle="arc3,rad=0.25",
                            shrinkA=18 if compact else 20, shrinkB=22 if compact else 24,
                            mutation_scale=10 if compact else 12),
            zorder=5
        )
        bubble = plt.Circle((x0, y0), 0.06 if compact else 0.08, fill=False,
                            linestyle=(0, (4, 4)), color="#d55", linewidth=1.6 if compact else 2)
        ax.add_patch(bubble)
        if step.back_value is not None:
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax.text(xm, ym + 0.05, f"{step.back_value}", color="#d55", fontsize=font_size, weight="bold")

    plt.title(title)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# =========================
# Tic-Tac-Toe & search code
# =========================
WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)
]

# Only Random (default) + Custom for ordering exploration
CURRENT_ORDER = random.sample(range(9), 9)

def _parse_order(text: str):
    try:
        vals = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
        if sorted(vals) != list(range(9)):
            return None
        return vals
    except Exception:
        return None

def _apply_ordering_mode(mode: str):
    global CURRENT_ORDER
    if mode == "Random (new each game)":
        CURRENT_ORDER = random.sample(range(9), 9)
    elif mode == "Custom (student)":
        vals = _parse_order(st.session_state.custom_order_text)
        if vals is not None:
            CURRENT_ORDER = vals

def ttt_winner(board):
    for a,b,c in WIN_LINES:
        if board[a] != " " and board[a] == board[b] == board[c]:
            return board[a]
    if " " not in board:
        return "draw"
    return None

def ttt_moves(board, ordered: bool = False):
    ms = [i for i,c in enumerate(board) if c == " "]
    if not ordered:
        return ms
    idx = {v:i for i,v in enumerate(CURRENT_ORDER)}
    return sorted(ms, key=lambda m: idx.get(m, 9999))

def ttt_minimax(board, player, alpha=-math.inf, beta=math.inf):
    w = ttt_winner(board)
    if w == "X": return 1, None
    if w == "O": return -1, None
    if w == "draw": return 0, None
    if player == "X":  # maximizing
        best, best_move = -math.inf, None
        for m in ttt_moves(board):
            board[m] = "X"
            val, _ = ttt_minimax(board, "O", alpha, beta)
            board[m] = " "
            if val > best:
                best, best_move = val, m
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best, best_move
    else:             # minimizing
        best, best_move = math.inf, None
        for m in ttt_moves(board):
            board[m] = "O"
            val, _ = ttt_minimax(board, "X", alpha, beta)
            board[m] = " "
            if val < best:
                best, best_move = val, m
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best, best_move

def _evaluate_board(board) -> int:
    """Heuristic at depth limit: + for X, − for O (10, 100 for threats)."""
    score = 0
    for a,b,c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        xs, os = line.count("X"), line.count("O")
        if xs and os:
            continue  # blocked
        if xs: score += 10**xs
        if os: score -= 10**os
    return score

def search_minimax(board, player, ordered=False):
    visited = 0
    def mm(turn):
        nonlocal visited
        visited += 1
        w = ttt_winner(board)
        if w == "X": return  1, None
        if w == "O": return -1, None
        if w == "draw": return 0, None
        if turn == "X":
            best, move = -math.inf, None
            for m in ttt_moves(board, ordered=ordered):
                board[m] = "X"
                v, _ = mm("O")
                board[m] = " "
                if v > best: best, move = v, m
            return best, move
        else:
            best, move = math.inf, None
            for m in ttt_moves(board, ordered=ordered):
                board[m] = "O"
                v, _ = mm("X")
                board[m] = " "
                if v < best: best, move = v, m
            return best, move
    t0 = time.perf_counter()
    val, move = mm(player)
    dt = time.perf_counter() - t0
    return val, move, {"algorithm":"Minimax","visited":visited,"prunes":0,"time_s":dt}

def search_alphabeta(board, player, ordered=False):
    visited, prunes = 0, 0
    def ab(turn, alpha, beta):
        nonlocal visited, prunes
        visited += 1
        w = ttt_winner(board)
        if w == "X": return  1, None
        if w == "O": return -1, None
        if w == "draw": return 0, None
        if turn == "X":
            value, move = -math.inf, None
            moves = ttt_moves(board, ordered=ordered)
            for i, m in enumerate(moves):
                board[m] = "X"
                v, _ = ab("O", alpha, beta)
                board[m] = " "
                if v > value: value, move = v, m
                alpha = max(alpha, value)
                if alpha >= beta:
                    prunes += len(moves) - (i+1)
                    break
            return value, move
        else:
            value, move = math.inf, None
            moves = ttt_moves(board, ordered=ordered)
            for i, m in enumerate(moves):
                board[m] = "O"
                v, _ = ab("X", alpha, beta)
                board[m] = " "
                if v < value: value, move = v, m
                beta = min(beta, value)
                if alpha >= beta:
                    prunes += len(moves) - (i+1)
                    break
            return value, move
    t0 = time.perf_counter()
    val, move = ab(player, -math.inf, math.inf)
    dt = time.perf_counter() - t0
    return val, move, {"algorithm":"Alpha–Beta","visited":visited,"prunes":prunes,"time_s":dt}

@contextmanager
def temp_order(order):
    global CURRENT_ORDER
    old = CURRENT_ORDER[:]
    CURRENT_ORDER = order[:]
    try:
        yield
    finally:
        CURRENT_ORDER = old

def choose_ai_move(board, ai_symbol, strategy, ordered=True):
    legal = ttt_moves(board, ordered=False)
    if not legal:
        return None, {"algorithm":strategy,"visited":0,"prunes":0,"time_s":0.0}
    side = ai_symbol
    if strategy == "Optimal (α–β minimax)":
        _, mv, met = search_alphabeta(board, side, ordered=ordered); return mv, met
    if strategy == "Minimax (no pruning)":
        _, mv, met = search_minimax(board, side, ordered=ordered);   return mv, met
    # Random
    t0 = time.perf_counter()
    mv = random.choice(legal)
    return mv, {"algorithm":"Random","visited":0,"prunes":0,"time_s":time.perf_counter()-t0}

def best_for_hint(board, for_player="X"):
    val, mv, _ = search_alphabeta(board, for_player, ordered=True)
    return val, mv

# =========================
# Build a static TTT tree
# =========================
def build_ttt_tree_nodes(board, side, max_depth=3, ordered=True):
    """
    Returns nodes dict for step visualizers.
    ROOT: side to move ("MAX" if X else "MIN").
    Every node carries (move, by): which 0..8 cell and who played it to REACH the node.
    """
    nodes: Dict[str, Node] = {}
    nodes["ROOT"] = Node("ROOT", "MAX" if side == "X" else "MIN", None, [], move=None, by=None)

    counter = {"n": 0}
    def new_id(prefix="N"):
        counter["n"] += 1
        return f"{prefix}{counter['n']}"

    def mk_leaf(val: int, last_move: Optional[int], by: Optional[str]):
        nid = new_id("L")
        nodes[nid] = Node(nid, "LEAF", val, [], move=last_move, by=by)
        return nid

    def util_of_winner(w):
        if w == "X": return 1000
        if w == "O": return -1000
        return 0 
    
    # debugging minimax issue
    legal_root = ttt_moves(board, ordered=ordered)
    if max_depth <= 0 or not legal_root:
        w = ttt_winner(board)
        if w is not None:               
            nodes["ROOT"].kind  = "LEAF"
            nodes["ROOT"].value = util_of_winner(w)
        else:                            
            nodes["ROOT"].kind  = "LEAF"
            nodes["ROOT"].value = _evaluate_board(board)
        return nodes

    def rec(board, player, depth, last_move: Optional[int], by: Optional[str]):
        w = ttt_winner(board)
        if w is not None:
            return mk_leaf(util_of_winner(w), last_move, by)
        if depth == 0:
            return mk_leaf(_evaluate_board(board), last_move, by)
        nid = new_id("T")
        kind = "MAX" if player == "X" else "MIN"
        children = []
        for m in ttt_moves(board, ordered=ordered):
            board[m] = player
            child = rec(board, "O" if player == "X" else "X", depth - 1, m, player)
            board[m] = " "
            children.append(child)
        nodes[nid] = Node(nid, kind, None, children, move=last_move, by=by)
        return nid

    for m in legal_root:
        board[m] = side
        child = rec(board, "O" if side == "X" else "X", max_depth - 1, m, side)
        board[m] = " "
        nodes["ROOT"].children.append(child)
    return nodes


MAX_COLLAPSED_LEAVES = 24

def build_collapsed_root_to_leaves(nodes_full: Dict[str, Node]) -> Tuple[Dict[str, Node], int, int]:
    leaves = [nid for nid, n in nodes_full.items() if n.kind == "LEAF"]
    shown = leaves if len(leaves) <= MAX_COLLAPSED_LEAVES else random.sample(leaves, MAX_COLLAPSED_LEAVES)
    root_kind = nodes_full["ROOT"].kind
    collapsed: Dict[str, Node] = {"ROOT": Node("ROOT", root_kind, None, shown[:])}
    for lid in shown:
        ln = nodes_full[lid]
        collapsed[lid] = Node(lid, "LEAF", ln.value, [], move=ln.move, by=ln.by)
    return collapsed, len(leaves), len(shown)

def rebuild_ttt_tree(board, ordered_flag: bool):
    """Recompute the step log for the viewer from the current position."""
    # If strategy is Random, do not build anything
    if st.session_state.ttt_tree_algo is None:
        st.session_state.ttt_tree_nodes = {}
        st.session_state.ttt_tree_steps = []
        st.session_state.ttt_tree_i = 0
        return

    side = "O" if board.count("X") > board.count("O") else "X"
    req_depth = st.session_state.ttt_tree_depth
    max_depth = min(req_depth, board.count(" "))
    if max_depth <= 0: max_depth = 1

    if max_depth <= 3:
        nodes = build_ttt_tree_nodes(board.copy(), side, max_depth=max_depth, ordered=ordered_flag)
        if st.session_state.ttt_tree_algo == "Alpha–Beta":
            steps = alphabeta_steps(nodes)
        else:
            steps = minimax_steps(nodes)
        st.session_state.ttt_tree_nodes = nodes
        st.session_state.ttt_tree_steps = steps
    else:
        st.session_state.ttt_tree_nodes = {}
        st.session_state.ttt_tree_steps = []
    st.session_state.ttt_tree_i = 0
    st.session_state.ttt_tree_auto_depth = max_depth

# =========================
# UI / Layout
# =========================
st.set_page_config(page_title="Minimax & Alpha-Beta — Step Tutorial", layout="wide")

# center the content
st.markdown(
    """
    <style>
      [data-testid="stAppViewContainer"] > .main .block-container { max-width: 60vw; margin: 0 auto; }
      section.main > div.block-container { max-width: 60vw; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; margin-bottom:0;'>Minimax & Alpha–Beta</h1>", unsafe_allow_html=True)
st.caption("Step through a sample tree; then play Tic-Tac-Toe against search agents and visualize the search & pruning.")

# Session state (toy)
if "nodes" not in st.session_state:         st.session_state.nodes = default_tree()
if "algo" not in st.session_state:          st.session_state.algo = "Minimax"
if "steps" not in st.session_state:         st.session_state.steps = minimax_steps(st.session_state.nodes)
if "i" not in st.session_state:             st.session_state.i = 0

# Session state (TTT)
if "ttt_board" not in st.session_state:     st.session_state.ttt_board = [" "] * 9
if "ttt_starter" not in st.session_state:   st.session_state.ttt_starter = "Student (X)"
if "ai_strategy" not in st.session_state:   st.session_state.ai_strategy = "Optimal (α–β minimax)"
if "ordering_mode" not in st.session_state: st.session_state.ordering_mode = "Random (new each game)"
if "custom_order_text" not in st.session_state: st.session_state.custom_order_text = "4,0,2,6,8,1,3,5,7"
if "last_metrics" not in st.session_state:  st.session_state.last_metrics = None

# Tree viewer state
if "ttt_tree_depth" not in st.session_state: st.session_state.ttt_tree_depth = 3
if "ttt_tree_algo" not in st.session_state:  st.session_state.ttt_tree_algo = "Alpha–Beta"  # will be synced to strategy below
if "ttt_tree_nodes" not in st.session_state: st.session_state.ttt_tree_nodes = {}
if "ttt_tree_steps" not in st.session_state: st.session_state.ttt_tree_steps = []
if "ttt_tree_i" not in st.session_state:     st.session_state.ttt_tree_i = 0

def _algo_from_strategy(strategy: str):
    if strategy == "Optimal (α–β minimax)": return "Alpha–Beta"
    if strategy == "Minimax (no pruning)":   return "Minimax"
    return None  # Random → no diagram

def _sync_algo_and_rebuild(board):
    st.session_state.ttt_tree_algo = _algo_from_strategy(st.session_state.ai_strategy)
    rebuild_ttt_tree(board, True)

tab_tree, tab_ttt = st.tabs(["Tree Tutorial", "Tic-Tac-Toe"])

# ---- TAB 1: toy tree stepper ----
with tab_tree:
    st.markdown("### Toy example — step through the algorithm")
    c1, c2 = st.columns([1,1])
    with c1:
        algo = st.radio("Algorithm", ["Minimax", "Alpha–Beta pruning"], horizontal=True)
    with c2:
        show_ab = st.checkbox("Show α/β on nodes", value=True)

    if algo != st.session_state.algo:
        st.session_state.algo = algo
        st.session_state.i = 0
        st.session_state.steps = minimax_steps(st.session_state.nodes) if algo == "Minimax" else alphabeta_steps(st.session_state.nodes)

    n1, n2, n3, n4 = st.columns([1,1,1,2])
    with n1:
        if st.button("⟵ Back", use_container_width=True):
            st.session_state.i = max(0, st.session_state.i - 1)
    with n2:
        if st.button("Next ⟶", use_container_width=True):
            st.session_state.i = min(len(st.session_state.steps) - 1, st.session_state.i + 1)
    with n3:
        if st.button("Reset", use_container_width=True):
            st.session_state.nodes = default_tree()
            st.session_state.i = 0
            st.session_state.steps = minimax_steps(st.session_state.nodes) if st.session_state.algo == "Minimax" else alphabeta_steps(st.session_state.nodes)
    with n4:
        st.write(f"Step {st.session_state.i+1} / {len(st.session_state.steps)}")

    step = st.session_state.steps[st.session_state.i]
    draw_tree(
        st.session_state.nodes, step,
        show_alpha_beta=(algo.startswith("Alpha") and show_ab),
        title=f"{st.session_state.algo} — Step {st.session_state.i+1}",
        compact=True
    )

    # >>> New: detailed step explainer (requested)
    with st.expander("What is happening in this step?", expanded=True):
        st.write(step.description)
        if step.frontier_eval:
            st.markdown("**Newly evaluated leaves:** " + ", ".join(f"{k}={v}" for k,v in step.frontier_eval.items()))
        if step.discovered_values:
            st.markdown("**Backed-up values:** " + ", ".join(f"{k}={v}" for k,v in step.discovered_values.items()))
        if algo.startswith("Alpha") and step.alpha_beta:
            def fmt_pair(a,b):
                a_str, b_str = _fmt_ab(a, b)
                return a_str, b_str
            parts = []
            for n,(a,b) in step.alpha_beta.items():
                as_, bs_ = fmt_pair(a,b)
                parts.append(f"{n}: α={as_}, β={bs_}")
            st.markdown("**Current α/β on path:** " + ", ".join(parts))
        if step.pruned_edges:
            pr = ", ".join([f"{u}→{v}" for (u,v) in step.pruned_edges])
            st.markdown(f"**Pruned edges:** {pr}")

# ---- TAB 2: TTT + visualization ----
with tab_ttt:
    st.markdown("### Tic-Tac-Toe: play and watch the search")
    st.caption("Cell numbers: **1 2 3 / 4 5 6 / 7 8 9**  (node labels show `m=k by P` → which move made that node)")

    ctrl = st.container()
    with ctrl:
        c_top = st.columns([1.4, 1.2, 1.2, 1.2, 1.0, 1.0])
        with c_top[0]:
            ordering_options = ["Random (new each game)", "Custom (student)"]
            st.session_state.ordering_mode = st.selectbox(
                "Move ordering (search)",
                ordering_options,
                index=ordering_options.index(st.session_state.ordering_mode)
                if st.session_state.ordering_mode in ordering_options else 0
            )
            st.caption(f"Current order: {CURRENT_ORDER}  (indices 0..8 → cells 1..9)")
        with c_top[1]:
            if st.session_state.ordering_mode == "Custom (student)":
                st.session_state.custom_order_text = st.text_input(
                    "Custom order 0..8",
                    st.session_state.custom_order_text,
                    help="Permutation of 0..8, e.g., 4,0,2,6,8,1,3,5,7"
                )
            if st.button("Apply ordering", use_container_width=True):
                _apply_ordering_mode(st.session_state.ordering_mode)
                _sync_algo_and_rebuild(st.session_state.ttt_board)
                st.success(f"Ordering set to: {CURRENT_ORDER}")
                st.rerun()
        with c_top[2]:
            t_depth = st.slider("Diagram depth", 1, 3, min(st.session_state.ttt_tree_depth, 3),
                                help="Depth limited to 3 for clarity.")

        with c_top[3]:
            starter = st.radio("Who starts?", ["Student (X)", "Computer (O)"], index=0, key="starter_radio")
        with c_top[4]:
            # choose strategy; diagram will follow this choice
            old_strategy = st.session_state.ai_strategy
            st.session_state.ai_strategy = st.selectbox(
                "Computer strategy",
                ["Optimal (α–β minimax)", "Minimax (no pruning)", "Random"],
                index=["Optimal (α–β minimax)", "Minimax (no pruning)", "Random"].index(st.session_state.ai_strategy)
            )
            if st.session_state.ai_strategy != old_strategy:
                _sync_algo_and_rebuild(st.session_state.ttt_board)
        with c_top[5]:
            if st.button("New game", key="btn_new_game", use_container_width=True):
                # reset board + metrics
                st.session_state.ttt_board = [" "] * 9
                st.session_state.ttt_starter = starter
                st.session_state.last_metrics = None

                # reset viewer state so the old tree does not linger
                st.session_state.ttt_tree_nodes = {}
                st.session_state.ttt_tree_steps = []
                st.session_state.ttt_tree_i = 0

                # keep ordering & depth, then rebuild and rerun
                _apply_ordering_mode(st.session_state.ordering_mode)
                st.session_state.ttt_tree_depth = t_depth
                _sync_algo_and_rebuild(st.session_state.ttt_board)
                st.rerun()


        # sync depth adjustment
        if t_depth != st.session_state.ttt_tree_depth:
            st.session_state.ttt_tree_depth = t_depth
            _sync_algo_and_rebuild(st.session_state.ttt_board)

    board = st.session_state.ttt_board

    # computer opens if chosen
    if st.session_state.ttt_starter == "Computer (O)" and board.count(" ") == 9:
        mv, metrics = choose_ai_move(board, ai_symbol="O", strategy=st.session_state.ai_strategy, ordered=True)
        if mv is not None:
            board[mv] = "O"
            st.session_state.last_metrics = metrics
            _sync_algo_and_rebuild(board)
            st.rerun()

    col_game, col_tree = st.columns([0.9, 1.4])

    with col_game:
        st.markdown("#### Board")
        st.caption("Click a square to place **X**. The computer replies as **O**.")
        clicked = None
        for r in range(3):
            cols = st.columns(3, gap="small")
            for cidx in range(3):
                i = 3*r + cidx
                label = board[i] if board[i] != " " else " "
                disabled = (board[i] != " ") or (ttt_winner(board) is not None)
                if cols[cidx].button(label if label != " " else " ", key=f"cell_{i}",
                                     use_container_width=True, disabled=disabled):
                    clicked = i
        if clicked is not None:
            if board[clicked] == " " and ttt_winner(board) is None:
                board[clicked] = "X"
                if ttt_winner(board) is None and " " in board:
                    mv, metrics = choose_ai_move(board, ai_symbol="O",
                                                 strategy=st.session_state.ai_strategy, ordered=True)
                    if mv is not None and board[mv] == " ":
                        board[mv] = "O"
                        st.session_state.last_metrics = metrics
                _sync_algo_and_rebuild(board)
            st.rerun()

        w = ttt_winner(board)
        if w == "X":   st.success("You win! 🎉")
        elif w == "O": st.error("Computer wins.")
        elif w == "draw": st.info("Draw.")

        if st.session_state.last_metrics:
            m = st.session_state.last_metrics
            st.markdown("#### Last AI move — metrics")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("AI algorithm", m["algorithm"])
            mc2.metric("Visited nodes", f"{m['visited']:,}")
            mc3.metric("Prunes", f"{m['prunes']:,}")
            mc4.metric("Time (ms)", f"{m['time_s']*1000:.2f}")

    with col_tree:
        st.markdown("#### Search diagram")

        # Build once on first load (or after you cleared on "New game")
        if not st.session_state.ttt_tree_nodes and st.session_state.ttt_tree_algo is not None:
            _sync_algo_and_rebuild(st.session_state.ttt_board)

        algo_for_diagram = st.session_state.ttt_tree_algo  # None if Random
        if algo_for_diagram is None:
            st.info("Random strategy: no search tree to show.")
        else:
            depth_to_build = min(st.session_state.ttt_tree_depth, st.session_state.ttt_board.count(" ")) or 1
            st.caption(f"Depth: {st.session_state.get('ttt_tree_auto_depth', depth_to_build)} (full tree)")

            nodes = st.session_state.ttt_tree_nodes
            steps = st.session_state.ttt_tree_steps
            root = nodes.get("ROOT") if nodes else None

            if not nodes:
                st.info("Preparing diagram… make a move or press **New game**.")
            elif root and root.kind == "LEAF":
                # This really is a terminal/limit position
                side_now = "O" if st.session_state.ttt_board.count("X") > st.session_state.ttt_board.count("O") else "X"
                if algo_for_diagram == "Alpha–Beta":
                    v, _m, _ = search_alphabeta(st.session_state.ttt_board.copy(), side_now, ordered=True)
                    st.info(f"Terminal/limit position — best value **{v}**; nothing to expand.")
                else:
                    v, _m, _ = search_minimax(st.session_state.ttt_board.copy(), side_now, ordered=True)
                    st.info(f"Terminal/limit position — best value **{v}**; nothing to expand.")
            elif steps:
                s_left, s_mid, s_right = st.columns([1,1,2])
                if s_left.button("⟵", key="ttt_tree_prev"):
                    st.session_state.ttt_tree_i = max(0, st.session_state.ttt_tree_i - 1)
                if s_mid.button("⟶", key="ttt_tree_next"):
                    st.session_state.ttt_tree_i = min(len(steps) - 1, st.session_state.ttt_tree_i + 1)
                s_right.write(f"Step {st.session_state.ttt_tree_i + 1} / {len(steps)}")

                step = steps[st.session_state.ttt_tree_i]
                draw_tree(
                    nodes,
                    step,
                    show_alpha_beta=(algo_for_diagram == "Alpha–Beta"),
                    title=f"{algo_for_diagram} — Step {st.session_state.ttt_tree_i+1}",
                    compact=True
                )
            else:
                st.info("Start a new game or apply a move ordering to build the tree.")

    # Legend lives in the Tic-Tac-Toe section (as requested)
    with st.expander("Legend & scoring"):
        st.markdown(
            "### Leaf values:\n"
            "• **+1000** = X win, **−1000** = O win, **0** = draw.\n"
            "• **Otherwise heuristic:**\n\n"
            "- Count each open line (row/col/diag) that contains only X’s or only O’s.\n"
            "- If the line has **1 X → +10**, **2 X’s → +100**.\n"
            "- If the line has **1 O → −10**, **2 O’s → −100**.\n\n"
            "Larger magnitude means stronger short-term threats.\n\n"
            "**Colors:** Yellow=current, Green=visited/evaluated, Blue=unvisited, Grey=pruned."
        )

    # ---- Benchmark block ----
    st.divider()
    st.markdown("### Benchmark: Minimax vs Alpha–Beta from this position")
    bench_left, bench_right = st.columns([1, 1])
    with bench_left:
        if st.button("Run benchmark", key="btn_bench", use_container_width=True):
            side_b = "O" if board.count("X") > board.count("O") else "X"
            v1, m1, mm = search_minimax(board.copy(), side_b, ordered=True)
            v2, m2, ab = search_alphabeta(board.copy(), side_b, ordered=True)
            st.session_state.bench = {"side": side_b, "mm": (v1, m1, mm), "ab": (v2, m2, ab)}
    with bench_right:
        if hasattr(st.session_state, "bench"):
            side_b = st.session_state.bench["side"]
            (v1, m1, mm), (v2, m2, abm) = st.session_state.bench["mm"], st.session_state.bench["ab"]
            st.write(f"Side to move: **{side_b}**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Minimax (no pruning)**")
                st.write(f"Move: {m1+1 if m1 is not None else '—'} | Value: {v1}")
                st.write(f"Visited: {mm['visited']:,} | Prunes: {mm['prunes']:,}")
                st.write(f"Time: {mm['time_s']*1000:.2f} ms")
            with c2:
                st.markdown("**Alpha–Beta**")
                st.write(f"Move: {m2+1 if m2 is not None else '—'} | Value: {v2}")
                st.write(f"Visited: {abm['visited']:,} | Prunes: {abm['prunes']:,}")
                st.write(f"Time: {abm['time_s']*1000:.2f} ms")

# Optional download (if present)
try:
    with open("Alpha-Beta Pruning - v2.pptx", "rb") as f:
        st.download_button("Download Alpha-Beta Pruning Guidance", f, file_name="Alpha-Beta Pruning - v2.pptx")
except FileNotFoundError:
    pass

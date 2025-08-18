#https://github.com/aimacode/aima-python/blob/master/games4e.py
#https://networkx.org/documentation/stable/reference/classes/digraph.html

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import math

@dataclass
class Node:
    id: str
    kind: str   
    value: Optional[int]  
    children: List[str] = field(default_factory=list)

def default_tree() -> Dict[str, Node]:
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
    nodes = {n.id: n for n in [ROOT, A, B, C, L1, L2, L3, M1, M2, M3, R1, R2, R3]}
    return nodes


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

# ~~~~~~~~algorithm section (min/max & alphabeta)~~~~~~~~~~~        
def minimax_steps(nodes: Dict[str, Node]) -> List[Step]:
    """DFS Minimax with a detailed step log (no pruning)."""
    steps: List[Step] = []
    discovered = {}
    frontier = {}
    ab = {}
    pruned = []

    def log(desc, current, back_edge=None, back_value=None):
        steps.append(
            Step(
                desc,
                current,
                discovered.copy(),
                frontier.copy(),
                ab.copy() if isinstance(ab, dict) else {},
                pruned.copy(),
                back_edge=back_edge,
                back_value=back_value
            )
        )

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
            if node.kind == "MAX":
                best = max(vals)
            else:
                best = min(vals)
            discovered[u] = best
            log(
            f"Back up {best} to {u} after exploring {ch}.",
            u,
            back_edge=(ch, u),     
            back_value=best
        )
        return discovered[u]

    dfs("ROOT")
    log("Minimax complete.", "ROOT")
    return steps

def alphabeta_steps(nodes: Dict[str, Node]) -> List[Step]:
    """DFS Alpha-Beta with detailed step log (including prunes)."""
    steps: List[Step] = []
    discovered = {}
    frontier = {}
    ab: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
    pruned: List[Tuple[str, str]] = []


    def log(desc, current, back_edge=None, back_value=None):
        steps.append(
            Step(
                desc,
                current,
                discovered.copy(),
                frontier.copy(),
                ab.copy() if isinstance(ab, dict) else {},
                pruned.copy(),
                back_edge=back_edge,
                back_value=back_value
            )
        )


    def dfs(u: str, alpha: int, beta: int) -> int:
        node = nodes[u]
        ab[u] = (alpha, beta)
        if node.kind == "LEAF":
            frontier[u] = node.value
            log(f"Evaluate leaf {u} = {node.value}.", u)
            return node.value

        if node.kind == "MAX":
            value = -10**9
            log(f"Enter MAX {u} with (α={alpha}, β={beta}).", u)
            for ch in node.children:
                val = dfs(ch, alpha, beta)
                value = max(value, val)
                discovered[u] = value
                alpha = max(alpha, value)
                ab[u] = (alpha, beta)
                log(f"Update {u}: value={value}, α={alpha}, β={beta} after {ch}.", u,    back_edge=(ch, u),back_value=value)
                if alpha >= beta:
                    # prune remaining children
                    idx = node.children.index(ch)
                    for pruned_child in node.children[idx+1:]:
                        pruned.append((u, pruned_child))
                        log(f"Prune edge {u}→{pruned_child} (α ≥ β).", u)
                    break
            return value
        else:  
            value = 10**9
            log(f"Enter MIN {u} with (α={alpha}, β={beta}).", u)
            for ch in node.children:
                val = dfs(ch, alpha, beta)
                value = min(value, val)
                discovered[u] = value
                beta = min(beta, value)
                ab[u] = (alpha, beta)
                log(f"Update {u}: value={value}, α={alpha}, β={beta} after {ch}.", u,back_edge=(ch, u),back_value=value )
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

# ~~~~~~~~ drawing section ~~~~~~~~~~~~~~~
def hierarchy_pos(G, root, width=2.8, vert_gap=0.28, vert_loc=1.0, xcenter=0.0, sibling_sep=0.0):
    """
    Position nodes in a hierarchy (root on top). Spreads siblings by subtree size
    and inserts a fixed gap (sibling_sep) between siblings at every level.
    """
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
def offset_along_line(p_from, p_to, dist):
    """Return a point starting at p_from and moving toward p_to by 'dist' (data units)."""
    x1, y1 = p_from
    x2, y2 = p_to
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L == 0:
        return p_from
    t = dist / L
    return (x1 + dx * t, y1 + dy * t)
def _fmt_ab(a, b):
    def f(x):
        if x == float("-inf"): return "-∞"
        if x == float("inf"):  return "∞"
        return str(x)
    return f(a), f(b)

def draw_tree(nodes: Dict[str, Node], step: Step, show_alpha_beta: bool, title: str):
    G = nx.DiGraph()
    for n in nodes.values():
        for ch in n.children:
            G.add_edge(n.id, ch)

    pos = hierarchy_pos(G, "ROOT", width=5.2, vert_gap=0.34, sibling_sep=0.12)

    colors, labels = [], {}
    for nid, n in nodes.items():
        if nid == step.current:
            colors.append("#ffd166")
        elif any((nid, ch) in step.pruned_edges for ch in nodes[nid].children):
            colors.append("#eeeeee")
        else:
            if nid in step.discovered_values or nid in step.frontier_eval:
                colors.append("#90ee90")
            else:
                colors.append("#c5d6ff")

        
        if n.kind == "LEAF":
            base = f"{nid}\n{n.value}"
        else:
            base = f"{nid} ({n.kind})"
            if nid in step.discovered_values:
                base += f"\n= {step.discovered_values[nid]}"

        if show_alpha_beta and nid in step.alpha_beta:
            a, b = step.alpha_beta[nid]
            a_str, b_str = _fmt_ab(a, b)
            base += f"\nα={a_str}, β={b_str}"

        labels[nid] = base
    pruned_set = set(step.pruned_edges)
    edge_colors = ["#cccccc" if (u, v) in pruned_set else "#333333" for u, v in G.edges()]
    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.margins(0.15)

    nx.draw(
        G, pos, with_labels=False, node_color=colors, edge_color=edge_colors,
        node_size=2000, arrows=False
    )
    nx.draw_networkx_labels(
        G, pos, labels, font_size=10,
        bbox=dict(facecolor="none", edgecolor="none", alpha=0.7, pad=0.5)
    )
    ax = plt.gca()
    if step.back_edge:
        child, parent = step.back_edge
        (x0, y0) = pos[child]
        (x1, y1) = pos[parent]

        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict( arrowstyle="-|>",lw=2,color="#d55", linestyle=(0, (6, 4)),connectionstyle="arc3,rad=0.25",
                shrinkA=20, shrinkB=24, mutation_scale=12
            ),
            zorder=5
        )
        bubble = plt.Circle((x0, y0), 0.08, fill=False,
                            linestyle=(0, (4, 4)), color="#d55", linewidth=2)
        ax.add_patch(bubble)

        if step.back_value is not None:
            xm, ym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax.text(xm, ym + 0.05, f"{step.back_value}",
                    color="#d55", fontsize=10, weight="bold")    
    plt.title(title)
    st.pyplot(fig)
    plt.close(fig)

st.set_page_config(page_title="Minimax & Alpha-Beta — Step Tutorial", layout="centered")

st.markdown(
    "<h1 style='text-align: center;'>Minimax & Alpha–Beta</h1>",
    unsafe_allow_html=True
)
st.caption("Click Next/Back to walk through the algorithm. Toggle between Minimax and Alpha–Beta.")

if "nodes" not in st.session_state:
    st.session_state.nodes = default_tree()
if "algo" not in st.session_state:
    st.session_state.algo = "Minimax"
if "steps" not in st.session_state:
    st.session_state.steps = minimax_steps(st.session_state.nodes)
if "i" not in st.session_state:
    st.session_state.i = 0

col1, col2 = st.columns([1,1])
with col1:
    algo = st.radio("Algorithm", ["Minimax", "Alpha–Beta pruning"], horizontal=True)
with col2:
    show_ab = st.checkbox("Show α/β on nodes", value=True)

if algo != st.session_state.algo:
    st.session_state.algo = algo
    st.session_state.i = 0
    if algo == "Minimax":
        st.session_state.steps = minimax_steps(st.session_state.nodes)
    else:
        st.session_state.steps = alphabeta_steps(st.session_state.nodes)

step = st.session_state.steps[st.session_state.i]

c1, c2, c3, c4 = st.columns([1,1,1,2])
with c1:
    if st.button("⟵ Back", use_container_width=True):
        st.session_state.i = max(0, st.session_state.i - 1)
        step = st.session_state.steps[st.session_state.i]
with c2:
    if st.button("Next ⟶", use_container_width=True):
        st.session_state.i = min(len(st.session_state.steps) - 1, st.session_state.i + 1)
        step = st.session_state.steps[st.session_state.i]
with c3:
    if st.button("Reset", use_container_width=True):
        st.session_state.nodes = default_tree()
        st.session_state.i = 0
        if st.session_state.algo == "Minimax":
            st.session_state.steps = minimax_steps(st.session_state.nodes)
        else:
            st.session_state.steps = alphabeta_steps(st.session_state.nodes)
        step = st.session_state.steps[0]
with c4:
    st.write(f"Step {st.session_state.i+1} / {len(st.session_state.steps)}")

draw_tree(st.session_state.nodes, step, show_alpha_beta=(algo.startswith("Alpha") and show_ab),
          title=f"{st.session_state.algo} — Step {st.session_state.i+1}")

with st.expander("What is happening in this step?", expanded=True):
    st.write(step.description)
    if step.frontier_eval:
        st.markdown("**Newly evaluated leaves:** " + ", ".join(f"{k}={v}" for k,v in step.frontier_eval.items()))
    if step.discovered_values:
        st.markdown("**Backed-up values:** " + ", ".join(f"{k}={v}" for k,v in step.discovered_values.items()))
    if algo.startswith("Alpha") and step.alpha_beta:
        trail = ", ".join(f"{n}: α={a if a is not None else '-∞'}, β={b if b is not None else '∞'}"
                          for n,(a,b) in step.alpha_beta.items())
        st.markdown("**Current α/β on path:** " + trail)
    if step.pruned_edges:
        pr = ", ".join([f"{u}→{v}" for (u,v) in step.pruned_edges])
        st.markdown(f"**Pruned edges:** {pr}")

st.info(
    "Tip: Switch to **Alpha–Beta pruning** and step through again to see where branches are cut off "
    "when α ≥ β, while the final root value stays the same as plain Minimax."
)
with open("Alpha-Beta Pruning - v2.pptx", "rb") as f:
    st.download_button("Download Alpha-Beta Pruning Guidance", f, file_name="Alpha-Beta Pruning - v2.pptx")

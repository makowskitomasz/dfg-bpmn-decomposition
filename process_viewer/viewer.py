"""Interactive zoomable visualization of DFG abstraction."""

from __future__ import annotations

import copy
from typing import Callable, Sequence, Type

import graphviz
import ipywidgets as widgets
import networkx as nx
import pm4py
from IPython.display import display
from networkx.algorithms.community import greedy_modularity_communities

from .subprocesses_labeler import ACTIVITY_IGNORE, name_subprocesses


def _build_base_graph(event_log: pm4py.objects.log.obj.EventLog) -> nx.DiGraph:
    print("📊 Calculating Base DFG...")
    dfg, start_acts, end_acts = pm4py.discover_dfg(event_log)

    graph = nx.DiGraph()
    for (src, tgt), freq in dfg.items():
        graph.add_edge(src, tgt, weight=freq)

    for node in list(graph.nodes()):
        if 'acts' not in graph.nodes[node]:
            graph.nodes[node]['acts'] = [node]
            graph.nodes[node]['label'] = str(node)
            graph.nodes[node]['type'] = 'atomic'

    graph.add_node("START_NODE", label="Start", type='start', acts=[])
    graph.add_node("END_NODE", label="End", type='end', acts=[])

    for act, freq in start_acts.items():
        if act in graph.nodes:
            graph.add_edge("START_NODE", act, weight=freq)

    for act, freq in end_acts.items():
        if act in graph.nodes:
            graph.add_edge(act, "END_NODE", weight=freq)

    return graph


def _merge_nodes(
    graph: nx.DiGraph,
    nodes: Sequence[str],
    new_id: str,
    labeler: Callable[[Sequence[str]], str],
) -> None:
    nodes_set = set(nodes)
    acts = []
    for n in nodes_set:
        acts.extend(graph.nodes[n].get('acts', [n]))
    graph.add_node(new_id, acts=acts, label=labeler(acts), type='group')

    for target in nodes_set:
        for pred in list(graph.predecessors(target)):
            if pred in nodes_set:
                continue
            w = graph[pred][target]['weight']
            if graph.has_edge(pred, new_id):
                graph[pred][new_id]['weight'] += w
            else:
                graph.add_edge(pred, new_id, weight=w)

    for source in nodes_set:
        for succ in list(graph.successors(source)):
            if succ in nodes_set:
                continue
            w = graph[source][succ]['weight']
            if graph.has_edge(new_id, succ):
                graph[new_id][succ]['weight'] += w
            else:
                graph.add_edge(new_id, succ, weight=w)

    for n in nodes_set:
        graph.remove_node(n)

    if graph.has_edge(new_id, new_id):
        graph.remove_edge(new_id, new_id)


class TopologicalAbstractor:
    def __init__(self, event_log, labeler: Callable[[Sequence[str]], str] = name_subprocesses):
        self.log = event_log
        self.labeler = labeler
        self.history = []

        self.G = _build_base_graph(event_log)

        self.history.append(self.get_current_state())
        self.run_abstraction()

    def get_current_state(self):
        return copy.deepcopy(self.G)

    def run_abstraction(self):
        """
        Algorithm:
        1. Priority 1: Strict Sequences (1-in, 1-out). Cleanest simplification.
        2. Priority 2: Absorb Weakest Node. 
           Find the node with the LOWEST total traffic (In+Out). 
           Merge it into its strongest neighbor.
        """
        iteration = 0
        
        while self.G.number_of_nodes() > 3: # Keep Start + End + at least 1 Group
            
            # --- STRATEGY A: Strict Sequences (Always First) ---
            candidates = []
            for u in self.G.nodes():
                if u in ["START_NODE", "END_NODE"]: continue
                
                # Check for 1 IN, 1 OUT pattern
                if self.G.out_degree(u) == 1:
                    succ = list(self.G.successors(u))[0]
                    if succ not in ["START_NODE", "END_NODE"] and self.G.in_degree(succ) == 1:
                         if u != succ:
                             candidates.append((u, succ, 'sequence'))

            # --- STRATEGY B: Absorb Weakest Node (Fallback) ---
            if not candidates:
                # 1. Calculate 'Total Volume' for every node
                node_weights = {}
                for n in self.G.nodes():
                    if n in ["START_NODE", "END_NODE"]: continue
                    
                    weight_in = sum(d['weight'] for _, _, d in self.G.in_edges(n, data=True))
                    weight_out = sum(d['weight'] for _, _, d in self.G.out_edges(n, data=True))
                    node_weights[n] = weight_in + weight_out
                
                # 2. Find the weakest node
                if node_weights:
                    weakest_node = min(node_weights, key=node_weights.get)
                    
                    # 3. Find weakest node's strongest neighbor (to merge with)
                    best_neighbor = None
                    max_conn_strength = -1
                    
                    # Check neighbors (Successors)
                    for succ in self.G.successors(weakest_node):
                         if succ in ["START_NODE", "END_NODE"]: continue
                         w = self.G[weakest_node][succ]['weight']
                         if w > max_conn_strength:
                             max_conn_strength = w
                             best_neighbor = succ
                    
                    # Check neighbors (Predecessors)
                    for pred in self.G.predecessors(weakest_node):
                         if pred in ["START_NODE", "END_NODE"]: continue
                         w = self.G[pred][weakest_node]['weight']
                         if w > max_conn_strength:
                             max_conn_strength = w
                             best_neighbor = pred
                    
                    if best_neighbor:
                        u, v = sorted([weakest_node, best_neighbor], key=lambda x: str(x))
                        candidates.append((u, v, 'weak_absorb'))

            if not candidates:
                break
            
            # --- EXECUTE MERGE ---
            u, v, method = candidates[0]
            
            # Create Group
            new_id = f"GRP_{iteration}"
            
            acts_u = self.G.nodes[u].get('acts', [u])
            acts_v = self.G.nodes[v].get('acts', [v])
            new_acts = acts_u + acts_v
            new_label = self.labeler(new_acts)
            
            self.G.add_node(new_id, acts=new_acts, label=new_label, type='group')
            
            # Reroute Edges
            # 1. Incoming to U or V -> New
            for target in [u, v]:
                for pred in list(self.G.predecessors(target)):
                    if pred == u or pred == v: continue 
                    w = self.G[pred][target]['weight']
                    if self.G.has_edge(pred, new_id):
                        self.G[pred][new_id]['weight'] += w
                    else:
                        self.G.add_edge(pred, new_id, weight=w)

            # 2. Outgoing from U or V -> New
            for source in [u, v]:
                for succ in list(self.G.successors(source)):
                    if succ == u or succ == v: continue 
                    w = self.G[source][succ]['weight']
                    if self.G.has_edge(new_id, succ):
                        self.G[new_id][succ]['weight'] += w
                    else:
                        self.G.add_edge(new_id, succ, weight=w)
            
            # Remove old nodes
            self.G.remove_node(u)
            self.G.remove_node(v)
            
            # Remove self-loops
            if self.G.has_edge(new_id, new_id):
                self.G.remove_edge(new_id, new_id)
            
            # Save State
            self.history.append(self.get_current_state())
            iteration += 1


class FlowProcessViewer:
    """Widget that lets users zoom into abstraction levels of a DFG."""

    def __init__(
        self,
        event_log,
        labeler: Callable[[Sequence[str]], str] | None = None,
        abstractor_cls: type = TopologicalAbstractor,
    ):
        self.engine = abstractor_cls(event_log, labeler=labeler or name_subprocesses)
        self.steps = self.engine.history
        self.max_step = len(self.steps) - 1
        
        self.out = widgets.Output()
        self.slider = widgets.IntSlider(
            value=0, min=0, max=self.max_step, step=1, 
            description='Zoom Level:', continuous_update=False
        )
        self.label_info = widgets.Label(value="Level 0: Detailed")
        self.slider.observe(self.on_slider_change, names='value')

        self.show_edges_label = abstractor_cls.__name__ in ("TopologicalAbstractor", "SCCModularityAbstractor")
        
    def render_graph(self, nx_graph):
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        dot.attr('node', fontname='Helvetica', fontsize='10')
        dot.attr('edge', fontname='Helvetica', fontsize='8', color='#555555')

        edges = list(nx_graph.edges(data=True))
        max_w = max([d['weight'] for u,v,d in edges]) if edges else 1

        for n, data in nx_graph.nodes(data=True):
            lbl = data.get('label', str(n))
            ntype = data.get('type', 'atomic')
            
            if ntype == 'start':
                dot.node(str(n), "Start", shape='circle', style='filled', fillcolor='#4caf50', fontcolor='white', width='0.6')
            elif ntype == 'end':
                dot.node(str(n), "End", shape='doublecircle', style='filled', fillcolor='#f44336', fontcolor='white', width='0.6')
            elif ntype == 'group':
                dot.node(str(n), lbl, shape='box3d', style='filled', fillcolor='#fff59d', penwidth='1')
            else:
                dot.node(str(n), lbl, shape='box', style='filled,rounded', fillcolor='#e3f2fd')

        for u, v, data in nx_graph.edges(data=True):
            w = data.get('weight', 1)
            width = 1.0 + (w / max_w) * 3.0
            edge_label = str(w) if self.show_edges_label else ""
            dot.edge(str(u), str(v), label=edge_label, penwidth=str(width))

        return dot

    def on_slider_change(self, change):
        self.update_view()

    def update_view(self):
        with self.out:
            self.out.clear_output(wait=True)
            step_idx = self.slider.value
            graph = self.steps[step_idx]
            self.label_info.value = f"Level {step_idx}: {graph.number_of_nodes()} Nodes"
            display(self.render_graph(graph))

    def show(self):
        ui = widgets.VBox([
            widgets.HBox([self.slider, self.label_info]),
            self.out
        ])
        display(ui)
        self.update_view()


class SCCModularityAbstractor:
    def __init__(self, event_log, labeler: Callable[[Sequence[str]], str] = name_subprocesses):
        self.log = event_log
        self.labeler = labeler
        self.history = []

        self.G = _build_base_graph(event_log)

        self.history.append(self.get_current_state())
        self.run_abstraction()

    def get_current_state(self):
        return copy.deepcopy(self.G)

    def _internal_weight(self, nodes: Sequence[str]) -> float:
        total = 0.0
        nodes_set = set(nodes)
        for u, v, data in self.G.edges(data=True):
            if u in nodes_set and v in nodes_set:
                total += data.get('weight', 1)
        return total

    def _best_pair_in_group(self, nodes: Sequence[str]) -> list[str] | None:
        nodes_set = set(nodes)
        best = None
        best_w = -1.0
        for u, v, data in self.G.edges(data=True):
            if u in nodes_set and v in nodes_set:
                w = data.get('weight', 1)
                if w > best_w:
                    best_w = w
                    best = [u, v]
        return best

    def _best_pair_avoiding_groups(self, nodes: Sequence[str]) -> list[str] | None:
        nodes_set = set(nodes)
        group_nodes = {n for n in nodes_set if self.G.nodes[n].get('type') == 'group'}
        candidates = []
        for u, v, data in self.G.edges(data=True):
            if u in nodes_set and v in nodes_set:
                w = data.get('weight', 1)
                uses_group = u in group_nodes or v in group_nodes
                candidates.append((uses_group, -w, [u, v]))
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    def _select_best_group(
        self, groups: Sequence[Sequence[str]], prefer_atomic: bool = True
    ) -> list[str] | None:
        candidates = []
        for group in groups:
            group = [n for n in group if n not in {"START_NODE", "END_NODE"}]
            if len(group) <= 1:
                continue
            has_group_nodes = any(self.G.nodes[n].get('type') == 'group' for n in group)
            candidates.append((has_group_nodes, len(group), self._internal_weight(group), group))
        if not candidates:
            return None
        # Prefer groups without existing group nodes, then smaller merges.
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3]

    def run_abstraction(self):
        iteration = 0
        cooldowns: dict[str, int] = {}

        while self.G.number_of_nodes() > 3:
            for node in list(cooldowns.keys()):
                cooldowns[node] -= 1
                if cooldowns[node] <= 0:
                    cooldowns.pop(node, None)

            # Phase 1: collapse SCCs to stabilize loops.
            sccs = [
                list(comp)
                for comp in nx.strongly_connected_components(self.G)
                if len(comp) > 1
                and "START_NODE" not in comp
                and "END_NODE" not in comp
            ]
            best_scc = self._select_best_group(
                self._filter_groups_for_cooldown(sccs, cooldowns, allow_fallback=True)
            )
            if best_scc:
                if len(best_scc) > 3:
                    pair = self._best_pair_avoiding_groups(best_scc)
                    if pair:
                        new_id = f"SCC_{iteration}"
                        _merge_nodes(self.G, pair, new_id, self.labeler)
                        iteration += 1
                        cooldowns[new_id] = 1
                        self.history.append(self.get_current_state())
                        continue
                new_id = f"SCC_{iteration}"
                _merge_nodes(self.G, best_scc, new_id, self.labeler)
                iteration += 1
                cooldowns[new_id] = 1
                self.history.append(self.get_current_state())
                continue

            # Phase 2: community detection via modularity on undirected view.
            undirected = nx.Graph()
            for u, v, data in self.G.edges(data=True):
                if u in {"START_NODE", "END_NODE"} or v in {"START_NODE", "END_NODE"}:
                    continue
                w = data.get('weight', 1)
                if undirected.has_edge(u, v):
                    undirected[u][v]['weight'] += w
                else:
                    undirected.add_edge(u, v, weight=w)

            if undirected.number_of_nodes() < 2:
                break

            communities = greedy_modularity_communities(undirected, weight='weight')
            groups = [list(c) for c in communities if len(c) > 1]
            best_group = self._select_best_group(
                self._filter_groups_for_cooldown(groups, cooldowns, allow_fallback=True)
            )
            if not best_group:
                break

            if len(best_group) > 3:
                pair = self._best_pair_avoiding_groups(best_group)
                if pair:
                    new_id = f"MOD_{iteration}"
                    _merge_nodes(self.G, pair, new_id, self.labeler)
                    iteration += 1
                    cooldowns[new_id] = 1
                    self.history.append(self.get_current_state())
                    continue

            new_id = f"MOD_{iteration}"
            _merge_nodes(self.G, best_group, new_id, self.labeler)
            iteration += 1
            cooldowns[new_id] = 1
            self.history.append(self.get_current_state())

    def _filter_groups_for_cooldown(
        self,
        groups: Sequence[Sequence[str]],
        cooldowns: dict[str, int],
        allow_fallback: bool = True,
    ) -> list[list[str]]:
        if not cooldowns:
            return [list(g) for g in groups]
        filtered = [
            [n for n in g if n not in {"START_NODE", "END_NODE"}]
            for g in groups
            if all(n not in cooldowns for n in g)
        ]
        if filtered:
            return filtered
        return [list(g) for g in groups] if allow_fallback else []

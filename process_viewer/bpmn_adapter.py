from __future__ import annotations
from typing import Any, Dict

import networkx as nx
from pm4py.objects.bpmn.obj import BPMN


def _bpmn_display_label(node) -> str:
    """Generate a display label for a BPMN node."""
    if getattr(node, "name", None):
        return str(node.name)

    # Otherwise label by type
    if isinstance(node, BPMN.StartEvent):
        return "Start"
    if isinstance(node, BPMN.EndEvent):
        return "End"
    if isinstance(node, BPMN.ExclusiveGateway):
        return "XOR"
    if isinstance(node, BPMN.ParallelGateway):
        return "AND"
    if isinstance(node, BPMN.InclusiveGateway):
        return "OR"
    if isinstance(node, BPMN.SubProcess):
        return "Subprocess"

    return "Task"


def _node_type(node: BPMN.BPMNNode) -> str:
    """Map PM4Py BPMN node to viewer-friendly type."""
    if isinstance(node, BPMN.StartEvent):
        return "start"
    if isinstance(node, BPMN.EndEvent):
        return "end"
    if isinstance(node, BPMN.ExclusiveGateway):
        return "xor_gateway"
    if isinstance(node, BPMN.ParallelGateway):
        return "and_gateway"
    if isinstance(node, BPMN.InclusiveGateway):
        return "or_gateway"
    if isinstance(node, BPMN.SubProcess):
        return "subprocess"
    return "atomic"  # tasks, activities


def bpmn_to_dfg_graph(bpmn_model) -> nx.DiGraph:
    G = nx.DiGraph()

    for node in bpmn_model.get_nodes():
        n_id = node.id
        n_type = _node_type(node)
        label = _bpmn_display_label(node)
        acts = [label] if n_type == "atomic" else []
        G.add_node(n_id, acts=acts, label=label, type=n_type)

    for flow in bpmn_model.get_flows():
        src = flow.source.id
        tgt = flow.target.id
        G.add_edge(src, tgt, weight=G[src][tgt]["weight"] + 1 if G.has_edge(src, tgt) else 1)

    return G


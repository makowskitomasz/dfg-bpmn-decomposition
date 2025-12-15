"""Interactive zoomable visualization of process trees."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

import graphviz
import ipywidgets as widgets
from IPython.display import display
import pm4py
from pm4py.objects.process_tree.obj import Operator, ProcessTree

from .subprocesses_labeler import ACTIVITY_IGNORE, name_subprocesses


class FlowProcessViewer:
    """Widget that lets users zoom into process tree fragments."""

    def __init__(
        self,
        log,
        min_collapse_size: int = 2,
        labeler: Callable[[Sequence[str]], str] | None = None,
    ):
        """Discover the process tree and prepare widget state.

        Args:
            log: PM4Py event log used to build the process tree.
            min_collapse_size: Minimum number of visible activities before a
                subtree gets collapsed at a given zoom level.
        """
        self.log = log
        self.tree: ProcessTree = pm4py.discover_process_tree_inductive(log)
        self.current_depth = 0
        self.max_depth = self._get_max_depth(self.tree)
        self.min_collapse_size = min_collapse_size
        self._leaf_cache = {}
        self._size_cache = {}
        self.labeler = labeler or name_subprocesses

        self.out = widgets.Output()
        self.btn_plus = widgets.Button(description="Zoom In", icon="search-plus")
        self.btn_minus = widgets.Button(description="Zoom Out", icon="search-minus")
        self.label = widgets.Label(value=f"Zoom Level: {self.current_depth} / {self.max_depth}")

        self.btn_plus.on_click(self.on_zoom_in)
        self.btn_minus.on_click(self.on_zoom_out)

    def _get_max_depth(self, node: ProcessTree, depth: int = 0) -> int:
        """Compute the deepest level underneath a node."""
        if not node.children:
            return depth
        return max(self._get_max_depth(child, depth + 1) for child in node.children)

    def _leaf_labels(self, node: ProcessTree) -> List[str]:
        """Collect all activity labels contained within the subtree."""
        cache_key = id(node)
        if cache_key in self._leaf_cache:
            return self._leaf_cache[cache_key]

        if not node.children:
            label = (node.label or "").strip()
            labels = [label] if label and label not in ACTIVITY_IGNORE else []
        else:
            labels = []
            for child in node.children:
                labels.extend(self._leaf_labels(child))

        self._leaf_cache[cache_key] = labels
        return labels

    def _subtree_size(self, node: ProcessTree) -> int:
        """Count how many meaningful activities live under the node."""
        cache_key = id(node)
        if cache_key in self._size_cache:
            return self._size_cache[cache_key]

        if not node.children:
            label = (node.label or "").strip()
            size = 1 if label and label not in ACTIVITY_IGNORE else 0
        else:
            size = sum(self._subtree_size(child) for child in node.children)

        self._size_cache[cache_key] = size
        return size

    def _is_collapsed(self, node: ProcessTree, current_depth: int, target_depth: int) -> bool:
        """Determine whether the subtree should be replaced with one box."""
        if not node.children:
            return False
        if current_depth < target_depth:
            return False
        return self._subtree_size(node) > self.min_collapse_size

    def _add_to_graph(self, g: graphviz.Digraph, node: ProcessTree, depth: int, target_depth: int) -> Tuple[str, str]:
        """Recursively add graph nodes/edges for the given process-tree node.

        Returns the entry and exit node ids of the rendered fragment so callers
        can connect parents and children regardless of whether the fragment was
        collapsed or expanded.
        """
        node_id = str(id(node))

        if self._is_collapsed(node, depth, target_depth):
            leaves = self._leaf_labels(node)
            clean_label = self.labeler(leaves)
            g.node(node_id, label=clean_label, shape="box3d", style="filled", fillcolor="#ffeb3b", fontname="Arial")
            return node_id, node_id

        if not node.children:
            if node.label is None:
                g.node(node_id, "", shape="point", width="0.1")
            else:
                g.node(node_id, node.label, shape="box", style="rounded,filled", fillcolor="#e3f2fd", fontname="Arial")
            return node_id, node_id

        child_boundaries = [self._add_to_graph(g, child, depth + 1, target_depth) for child in node.children]

        start_id = f"start_{node_id}"
        end_id = f"end_{node_id}"

        if node.operator == Operator.SEQUENCE:
            for i in range(len(child_boundaries) - 1):
                curr_end = child_boundaries[i][1]
                next_start = child_boundaries[i + 1][0]
                g.edge(curr_end, next_start)
            return child_boundaries[0][0], child_boundaries[-1][1]

        if node.operator in [Operator.XOR, Operator.PARALLEL]:
            gw_label = "×" if node.operator == Operator.XOR else "+"
            g.node(start_id, gw_label, shape="diamond", style="filled", fillcolor="#ffe0b2", height="0.3", width="0.3", fixedsize="true", fontsize="10")
            g.node(end_id, gw_label, shape="diamond", style="filled", fillcolor="#ffe0b2", height="0.3", width="0.3", fixedsize="true", fontsize="10")
            for c_start, c_end in child_boundaries:
                g.edge(start_id, c_start)
                g.edge(c_end, end_id)
            return start_id, end_id

        if node.operator == Operator.LOOP:
            body_s, body_e = child_boundaries[0]
            redo_s, redo_e = child_boundaries[1] if len(child_boundaries) > 1 else (None, None)
            if redo_s:
                g.edge(body_e, redo_s, label="redo", fontsize="8")
                g.edge(redo_e, body_s)
            else:
                g.edge(body_e, body_s, label="loop", fontsize="8")
            return body_s, body_e

        return node_id, node_id

    def render(self) -> None:
        """Draw the current zoom level and show it inside the output widget."""
        with self.out:
            self.out.clear_output(wait=True)
            g = graphviz.Digraph(format="png")
            g.attr(rankdir="LR")
            g.attr("node", fontname="Helvetica")
            g.node("START", "Start", shape="circle", style="filled", fillcolor="#4caf50", fontcolor="white", width="0.6")
            g.node("END", "End", shape="doublecircle", style="filled", fillcolor="#f44336", fontcolor="white", width="0.6")
            root_s, root_e = self._add_to_graph(g, self.tree, 0, self.current_depth)
            g.edge("START", root_s)
            g.edge(root_e, "END")
            display(g)

    def on_zoom_in(self, _button) -> None:
        """Increase the zoom level if we haven't reached the maximum."""
        if self.current_depth < self.max_depth:
            self.current_depth += 1
            self.update_ui()

    def on_zoom_out(self, _button) -> None:
        """Decrease the zoom level while keeping it non-negative."""
        if self.current_depth > 0:
            self.current_depth -= 1
            self.update_ui()

    def update_ui(self) -> None:
        """Refresh the label and re-render the graph for the current depth."""
        self.label.value = f"Zoom Level: {self.current_depth} / {self.max_depth}"
        self.render()

    def show(self) -> None:
        """Display the interactive controls and initial drawing."""
        controls = widgets.HBox([self.btn_minus, self.label, self.btn_plus])
        display(widgets.VBox([controls, self.out]))
        self.render()

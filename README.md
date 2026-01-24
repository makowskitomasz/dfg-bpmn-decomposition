# dfg-bpmn-decomposition

## Project Goal
Process Mining Project 2025/2026: automatically decompose complex Directly-Follows Graphs (DFGs) into smaller subprocesses, label each fragment (heuristics or GPT), and present them through an interactive zoomable viewer. The same approach will later be extended to BPMN models (part II).

## How to Run
1. Create/update the virtual environment and install dependencies:
   ```bash
   make build
   ```
2. Execute notebooks (e.g., `notebooks/milestone1.ipynb`) inside the venv.
3. To enable GPT labeling, create `.env` with `OPENAI_API_KEY=...`. Otherwise the viewer falls back to heuristic labels.

## Repository Structure
- `data/` – sample event logs plus `subprocesses_labels.json` cache for GPT-generated names.
- `process_viewer/` – reusable package with the zoomable viewer and labeling utilities.
  - `viewer.py` – `DFGFlowProcessViewer` widget with depth-based collapsing.
  - `subprocesses_labeler.py` – heuristics + optional GPT labeler with caching.
- `notebooks/`
  - `overview.ipynb` – introductory material (log loading, basic discovery).
  - `milestone1.ipynb` – part I of the project with the viewer wired in.
- `PM_2025_Process_Mining_Project.ipynb` – original combined notebook (still runnable via `make run`).
- `Makefile` – targets for build/run/format/test/clean based on the venv.
- `pyproject.toml` – dependency list and package configuration.

## Milestone 1
This milestone demonstrates DFG decomposition with an interactive zoomable viewer and two abstraction strategies.

### DFGFlowProcessViewer
`DFGFlowProcessViewer` builds the initial DFG from the event log and stores a history of abstraction steps. The widget exposes a zoom slider that renders any step as a Graphviz diagram. It takes an `abstractor_cls` so you can switch between strategies without changing the UI code.

### TopologicalAbstractor
`TopologicalAbstractor` is a simple baseline driven by local structure. It first collapses strict sequences where a node has exactly one predecessor and one successor, which preserves clear linear flow. When no such sequences remain, it finds the node with the lowest total traffic (sum of incoming and outgoing edge weights) and merges it into its strongest neighbor by edge weight. This reduces noise early and keeps high-traffic hubs intact longer. Each merge produces a new history level so the viewer can zoom through intermediate abstractions.

### SCCModularityAbstractor
`SCCModularityAbstractor` focuses on structural cohesion. It first collapses strongly connected components to stabilize loops and recurring cycles. If no SCCs remain, it detects communities using modularity on an undirected, weighted view of the DFG, which tends to group dense regions. To avoid big jumps, when a detected group has more than three nodes it merges only the strongest internal pair (highest-weight edge). It also prefers merges that do not include previously merged group nodes when possible, so multiple groups can emerge in parallel. This produces finer-grained levels and keeps the abstraction smooth instead of collapsing large regions at once.

### Subprocess Labeling
`name_subprocesses_with_gpt` calls `gpt-5-nano` and caches results in `data/subprocesses_labels.json`. The cache key is order-independent, so the same set of activities maps to one label. Labels are requested in Title Case (e.g., "Repair Cycle").  
`name_subprocesses` is the offline fallback that builds a label from activity text (first/last + common theme word).

## Current Status
- Repair-example log is ingested, converted to an event log, and a process tree is discovered via Inductive Miner.
- `DFGFlowProcessViewer` renders the tree as a graph, allowing zooming in/out by collapsing/expanding subtrees.
- Subprocess labeling works in two modes: heuristic (offline) and GPT-powered (`gpt-5-nano`), with caching of remote responses.
- Notebooks are split into overview and milestone parts, so the project presentation can gradually drill into the solution.
- Tooling (Makefile + pyproject) prepares a reproducible venv, formatting, and testing hooks.

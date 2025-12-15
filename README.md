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
  - `viewer.py` – `FlowProcessViewer` widget with depth-based collapsing.
  - `subprocesses_labeler.py` – heuristics + optional GPT labeler with caching.
- `notebooks/`
  - `overview.ipynb` – introductory material (log loading, basic discovery).
  - `milestone1.ipynb` – part I of the project with the viewer wired in.
- `PM_2025_Process_Mining_Project.ipynb` – original combined notebook (still runnable via `make run`).
- `Makefile` – targets for build/run/format/test/clean based on the venv.
- `pyproject.toml` – dependency list and package configuration.

## Current Status
- Repair-example log is ingested, converted to an event log, and a process tree is discovered via Inductive Miner.
- `FlowProcessViewer` renders the tree as a graph, allowing zooming in/out by collapsing/expanding subtrees.
- Subprocess labeling works in two modes: heuristic (offline) and GPT-powered (`gpt-5-nano`), with caching of remote responses.
- Notebooks are split into overview and milestone parts, so the project presentation can gradually drill into the solution.
- Tooling (Makefile + pyproject) prepares a reproducible venv, formatting, and testing hooks.

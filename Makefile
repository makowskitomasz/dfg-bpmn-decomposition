.PHONY: run build format test clean check-graphviz

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

build: check-graphviz $(PYTHON)
	$(PIP) install --upgrade pip
	$(PIP) install -e .

$(PYTHON):
	python3 -m venv $(VENV)

check-graphviz:
	@command -v dot >/dev/null 2>&1 || (echo "Graphviz 'dot' executable not found. Install Graphviz (e.g. 'brew install graphviz') and re-run make." >&2; exit 1)

clean:
	rm -rf $(VENV)

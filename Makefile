.PHONY: install install-verify verify lint figures clean

install:          ## install the full pinned environment
	pip install -r requirements.txt

install-verify:   ## minimal deps for the smoke test
	pip install -r requirements-verify.txt

verify:           ## re-derive headline numbers from the committed receipts (no data needed)
	python verify.py

lint:             ## static checks
	ruff check scripts verify.py

figures:          ## regenerate every figure (needs the acquired data)
	for f in scripts/figures/make_*.py; do python "$$f"; done

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

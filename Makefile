.PHONY: lint format test release

lint:
	ruff check . --fix

format:
	black .

test:
	pytest -v --capture=no

release:
	python -m build
	twine check dist/*
	twine upload dist/*
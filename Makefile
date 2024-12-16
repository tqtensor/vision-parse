.PHONY: lint format test

lint:
	ruff check . --fix

format:
	black .

test:
	pytest -v --capture=no
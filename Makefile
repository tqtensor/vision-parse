.PHONY: lint format test build release tag format-nb

lint:
	ruff check . --fix

format-nb:
	black --ipynb **/**/*.ipynb

format: format-nb
	black .

test:
	pytest -v --capture=no

build:
	python -m build
	twine check dist/*

tag:
	@read -p "Enter version (e.g., 0.1.0): " version; \
	git tag -a "v$$version" -m "Release v$$version"
	git push --tags

release: build tag
	@echo "Release workflow will be triggered by the tag push"
	@echo "Distribution files are available in ./dist directory"

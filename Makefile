.PHONY: lint format test release tag format-nb

lint:
	ruff check . --fix

format-nb:
	black --ipynb examples/*.ipynb

format: format-nb
	black .

test:
	pytest -v --capture=no

release: tag
	@echo "Release workflow will be triggered by the tag push"

tag:
	@read -p "Enter version (e.g., 0.1.0): " version; \
	git tag -a "v$$version" -m "Release v$$version"
	git push --tags
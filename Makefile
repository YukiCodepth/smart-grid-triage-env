.PHONY: install format lint test validate

install:
	pip install -r requirements.txt
	pip install pytest black mypy

format:
	black env/ inference.py

lint:
	mypy env/ inference.py --ignore-missing-imports

validate:
	# Simulates the Hackathon Phase 1 check
	python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"
	# Add the official openenv validate command here once installed
	# openenv validate

docker-test:
	docker build -t smartgrid-env .
	docker run --rm smartgrid-env python inference.py

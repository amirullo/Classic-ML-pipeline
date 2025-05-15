.PHONY: lint fix test coverage docker-build docker-push

lint:
	ruff check .

fix:
	ruff check . --fix

format:
	ruff format .

test:
	pytest -v

coverage:
	pytest --cov=. --cov-report=term --cov-report=xml

docker-build:
	docker build -t $(DOCKER_USER)/fastapi-ml-app:latest .

docker-push:
	docker push $(DOCKER_USER)/fastapi-ml-app:latest

.PHONY: up module2 train-module2 test-module2 frontend-dev
up:
	docker compose up --build
module2:
	docker compose -f docker-compose.module2-only.yml up --build
train-module2:
	cd services/module2-impaction-boneloss && python -m training.train_fdi data/fdi_segmentation/dataset.yaml
test-module2:
	cd services/module2-impaction-boneloss && python -m pytest tests/ -q
frontend-dev:
	cd frontend && npm ci && npm run dev

.PHONY: up module2 train-module2 export-module2 test-module2 frontend-dev

up:
	docker compose up --build

module2:
	docker compose -f docker-compose.module2-only.yml up --build

train-module2:
	cd services/module2-impaction-boneloss && python training/train_all.py

export-module2:
	cd services/module2-impaction-boneloss && python training/export_savedmodel.py

test-module2:
	cd services/module2-impaction-boneloss && pytest tests/ -q

frontend-dev:
	cd frontend && npm install && npm run dev

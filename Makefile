.PHONY: help up down restart logs clean test seed backup restore validate deploy reset

help:
	@echo "Agentic AI Agent Platform - Make Commands"
	@echo "==========================================="
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make restart    - Restart all services"
	@echo "  make logs       - View logs"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make test       - Run all tests"
	@echo "  make seed       - Seed database with demo data"
	@echo "  make reset      - Reset database (delete all data)"
	@echo "  make backup     - Backup database"
	@echo "  make restore    - Restore database from backup"
	@echo "  make validate   - Validate simulation accuracy"
	@echo "  make deploy     - Deploy to production"
	@echo "  make dev        - Start development environment"
	@echo "  make shell-api  - Open shell in API container"
	@echo "  make shell-db   - Open PostgreSQL shell"
	@echo "  make migrate    - Run database migrations"
	@echo "  make golden     - Run golden test suite"

up: fix-perms
	@echo "Starting Agentic AI Platform..."
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@bash scripts/project/update_container_labels.sh 2>/dev/null || true
	@echo "Platform is running!"
	@echo "Dashboard: http://localhost:8501"
	@echo "API: http://localhost:8000"
	@echo "Grafana: http://localhost:3000"

fix-perms:
	@echo "Fixing file permissions for Docker..."
	@chmod -R a+r src/ config/ data/ scripts/ tests/ 2>/dev/null || true
	@find src/ config/ data/ scripts/ tests/ -type d -exec chmod a+rx {} \; 2>/dev/null || true

down:
	@echo "Stopping Agentic AI Platform..."
	docker-compose down

restart:
	@echo "Restarting Agentic AI Platform..."
	docker-compose restart
	@sleep 10
	@bash scripts/project/update_container_labels.sh 2>/dev/null || true

logs:
	docker-compose logs -f

clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v
	rm -rf logs/*.log
	rm -rf data/experiments/*
	@echo "Cleanup complete!"

test:
	@echo "Running unit tests..."
	docker-compose run --rm api pytest tests/unit -v
	@echo "Running integration tests..."
	docker-compose run --rm api pytest tests/integration -v
	@echo "Running golden test suite..."
	docker-compose run --rm api python tests/golden/test_runner.py

seed:
	@echo "Seeding database with demo data..."
	docker-compose run --rm api python scripts/seed_data.py
	@echo "Database seeded successfully!"

reset:
	@echo "⚠️  WARNING: This will DELETE ALL DATA from the database!"
	@echo "This action cannot be undone!"
	@echo ""
	docker-compose exec -T api python scripts/reset_database.py
	@echo "Database reset complete!"

backup:
	@echo "Creating database backup..."
	@mkdir -p backups
	@TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	docker-compose exec postgres pg_dump -U $${POSTGRES_USER:-agentic} $${POSTGRES_DB:-agentic} > backups/backup_$$TIMESTAMP.sql
	@echo "Backup created: backups/backup_$$(date +%Y%m%d_%H%M%S).sql"

restore:
	@echo "Restoring database from latest backup..."
	@LATEST=$$(ls -t backups/*.sql | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No backup found!"; \
	else \
		echo "Restoring from $$LATEST"; \
		docker-compose exec -T postgres psql -U $${POSTGRES_USER:-agentic} $${POSTGRES_DB:-agentic} < $$LATEST; \
		echo "Database restored successfully!"; \
	fi

validate:
	@echo "Validating simulation accuracy against historical data..."
	docker-compose run --rm api python scripts/validate_simulation.py
	@echo "Validation complete! Check logs for accuracy metrics."

deploy:
	@echo "WARNING: This will deploy to production!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	@echo "Running golden test suite..."
	@make golden
	@echo "Building production images..."
	docker-compose build --no-cache
	@echo "Deployment complete!"

deploy-ci:
	@echo "Running golden test suite..."
	@make golden
	@echo "Building production images..."
	docker-compose build --no-cache
	@echo "Deployment complete!"

dev:
	@echo "Starting development environment..."
	docker-compose up -d postgres redis
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Starting API server..."
	uvicorn src.api.main:app --reload &
	@echo "Starting dashboard..."
	streamlit run dashboard/app.py &
	@echo "Development environment ready!"

shell-api:
	docker-compose exec api /bin/bash

shell-db:
	docker-compose exec postgres psql -U agentic

migrate:
	@echo "Running database migrations..."
	docker-compose run --rm api alembic upgrade head
	@echo "Migrations complete!"

golden:
	@echo "Running golden test suite..."
	python3 tests/golden/test_runner.py --strict
	@if [ $$? -eq 0 ]; then \
		echo "✅ All golden tests passed!"; \
	else \
		echo "❌ Golden tests failed! Deployment blocked."; \
		exit 1; \
	fi

health:
	@echo "Checking service health..."
	@curl -s http://localhost:8000/health | python -m json.tool
	@echo "Dashboard status:"
	@curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "Not responding"

logs-%:
	docker-compose logs -f $*

restart-%:
	docker-compose restart $*

experiment:
	@echo "Running experiment..."
	docker-compose run --rm api python scripts/run_experiments.py

report:
	@echo "Generating performance report..."
	docker-compose run --rm api python scripts/generate_report.py
	@echo "Report saved to data/reports/"
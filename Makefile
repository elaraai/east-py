.PHONY: all install test test-east-py test-east-py-std test-east-py-io test-integration lint typecheck clean build publish help

# Default target
all: install

# Install dependencies for all packages
install:
	uv sync

# Run all tests
test: test-east-py test-east-py-std test-east-py-io

# Run east-py tests
test-east-py:
	uv run --package east-py pytest packages/east-py/tests --durations=0

# Run east-py-std tests
test-east-py-std:
	uv run --package east-py-std pytest packages/east-py-std/tests

# Run east-py-io tests (--ignore-glob handles empty test dirs gracefully)
test-east-py-io:
	uv run --package east-py-io pytest packages/east-py-io/tests || [ $$? -eq 5 ]

# Run integration tests with Docker services (east-py-io only)
test-integration:
	cd packages/east-py-io && docker-compose up -d
	sleep 5  # Wait for services to start
	uv run --package east-py-io pytest packages/east-py-io/tests
	cd packages/east-py-io && docker-compose down -v

# Run linter on all packages
lint:
	uv run ruff check packages/

# Auto-fix linting issues
lint-fix:
	uv run ruff check --fix packages/

# Format code
format:
	uv run ruff format packages/

# Type check all packages
typecheck:
	cd packages/east-py && uv run mypy east
	cd packages/east-py-std && uv run mypy east_py_std
	cd packages/east-py-io && uv run mypy east_py_io

# Run all quality checks (lint + typecheck + test)
check: lint typecheck test

# Clean build artifacts and cache
clean:
	rm -rf .venv uv.lock
	rm -rf build/ dist/
	find packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true

# Build distribution packages
build:
	uv build --package east-py
	uv build --package east-py-std
	uv build --package east-py-io

# Publish packages to PyPI
publish:
	uv publish --package east-py
	uv publish --package east-py-std
	uv publish --package east-py-io

# Start Docker services for east-py-io development
services:
	docker-compose -f packages/east-py-io/docker-compose.yml up -d

# Stop Docker services
services-down:
	docker-compose -f packages/east-py-io/docker-compose.yml down -v

# View service logs
services-logs:
	docker-compose -f packages/east-py-io/docker-compose.yml logs -f

# Check service status
services-status:
	docker-compose -f packages/east-py-io/docker-compose.yml ps

# Help target
help:
	@echo "Available targets:"
	@echo "  install            - Install dependencies for all packages"
	@echo "  test               - Run all tests"
	@echo "  test-east-py       - Run east-py tests"
	@echo "  test-east-py-std   - Run east-py-std tests"
	@echo "  test-east-py-io    - Run east-py-io tests"
	@echo "  test-integration   - Run integration tests with Docker"
	@echo "  lint               - Run linter on all packages"
	@echo "  lint-fix           - Auto-fix linting issues"
	@echo "  format             - Format code"
	@echo "  typecheck          - Type check all packages"
	@echo "  check              - Run all quality checks"
	@echo "  clean              - Clean build artifacts and cache"
	@echo "  build              - Build distribution packages"
	@echo "  publish            - Publish packages to PyPI"
	@echo "  services           - Start Docker services"
	@echo "  services-down      - Stop Docker services"
	@echo "  services-logs      - View service logs"
	@echo "  services-status    - Check service status"
	@echo "  help               - Show this help message"

.PHONY: install install-cli test test-east-py-datascience lint format typecheck check clean build services-up services-down help

# Install dependencies
install:
	uv sync

# Install east-py command globally
install-cli:
	uv tool install --force --editable packages/east-py-cli \
		--with ./packages/east-py \
		--with ./packages/east-py-std \
		--with ./packages/east-py-io \
		--with ./packages/east-py-datascience

# Run all tests (per-package due to fixture isolation, but run all even if some fail)
test:
	@cd packages/east-py-datascience && npm run test:export
	@exit_code=0; \
	uv run --package east-py pytest packages/east-py/tests -v --durations=0 || exit_code=1; \
	uv run --package east-py-std pytest packages/east-py-std/tests -v --durations=0 || exit_code=1; \
	uv run --package east-py-io pytest packages/east-py-io/tests -v --durations=0 || exit_code=1; \
	uv run --package east-py-datascience pytest packages/east-py-datascience/tests -v --durations=0 || exit_code=1; \
	exit $$exit_code

# Run linter
lint:
	uv run ruff check packages/
	cd packages/east-py-datascience && npm run lint

# Format code
format:
	uv run ruff format packages/

# Type check
typecheck:
	cd packages/east-py && uv run mypy east
	cd packages/east-py-std && uv run mypy east_py_std
	cd packages/east-py-io && uv run mypy east_py_io
	cd packages/east-py-cli && uv run mypy east_py_cli
	cd packages/east-py-datascience && uv run mypy src/east_py_datascience

# Run all quality checks
check: lint typecheck test

# Clean build artifacts
clean:
	rm -rf .venv uv.lock build/ dist/
	find packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	find packages -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true

# Build packages
build:
	uv build --package east-py
	uv build --package east-py-std
	uv build --package east-py-io
	uv build --package east-py-cli
	cd packages/east-py-datascience && npm run build
	uv build --package east-py-datascience

# Start Docker services (for integration tests)
services-up:
	docker-compose -f packages/east-py-io/docker-compose.yml up -d

# Stop Docker services
services-down:
	docker-compose -f packages/east-py-io/docker-compose.yml down -v

# Help
help:
	@echo "install      - Install dependencies (uv sync)"
	@echo "test         - Run all tests"
	@echo "lint         - Run linter"
	@echo "format       - Format code"
	@echo "typecheck    - Type check"
	@echo "check        - Run lint + typecheck + test"
	@echo "clean        - Clean build artifacts"
	@echo "build        - Build packages"
	@echo "services-up  - Start Docker services"
	@echo "services-down - Stop Docker services"

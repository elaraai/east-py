.PHONY: install install-cli test test-east-py test-east-py-std test-east-py-io test-east-py-datascience test-export lint lint-headers lint-headers-fix format typecheck check clean build build-cython clean-cython services-up services-down help

# Install dependencies
install:
	@cd packages/east-py-datascience && npm install
	uv sync --all-extras --all-packages

# Update @elaraai dependencies (including transitive)
update:
	@cd packages/east-py-datascience && $(NVM) npm update @elaraai/east @elaraai/east-node-std

# Install east-py command globally (also syncs local .venv)
install-cli:
	uv sync --all-extras --all-packages
	uv tool install --force --editable packages/east-py-cli \
		--with-editable ./packages/east-py \
		--with-editable ./packages/east-py-std \
		--with-editable ./packages/east-py-io \
		--with-editable ./packages/east-py-datascience

# Export test IR from TypeScript packages
test-export:
	@cd packages/east-py-datascience && npm run test:export

# Run tests for individual packages
test-east-py:
	uv run --package east-py pytest packages/east-py/tests -v --durations=0

test-east-py-std:
	uv run --package east-py-std pytest packages/east-py-std/tests -v --durations=0

test-east-py-io:
	uv run --package east-py-io pytest packages/east-py-io/tests -v --durations=0

test-east-py-datascience:
	@cd packages/east-py-datascience && npm run test:export
	uv run --package east-py-datascience pytest packages/east-py-datascience/tests -v --durations=0

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
lint: lint-headers
	uv run ruff check packages/
	cd packages/east-py-datascience && npm run lint

# Check license headers (fails if any files need updating)
lint-headers:
	uv run python scripts/check_headers.py

# Fix license headers
lint-headers-fix:
	uv run python scripts/check_headers.py --fix

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

# Build Cython acceleration modules
build-cython:
	uv run --package east-py python packages/east-py/scripts/build_cython.py

# Clean Cython build artifacts
clean-cython:
	find packages/east-py -name "*.so" -delete
	find packages/east-py -name "*_cy.c" -delete

# Help
help:
	@echo "install           - Install dependencies (uv sync)"
	@echo "test              - Run all tests"
	@echo "test-east-py      - Run east-py tests only"
	@echo "test-east-py-std  - Run east-py-std tests only"
	@echo "test-east-py-io   - Run east-py-io tests only"
	@echo "test-east-py-datascience - Run east-py-datascience tests (export IR + pytest)"
	@echo "test-export       - Export test IR from TypeScript packages"
	@echo "lint              - Run linter (includes license header check)"
	@echo "lint-headers      - Check license headers only"
	@echo "lint-headers-fix  - Add missing license headers"
	@echo "format            - Format code"
	@echo "typecheck         - Type check"
	@echo "check             - Run lint + typecheck + test"
	@echo "clean             - Clean build artifacts"
	@echo "build             - Build packages"
	@echo "services-up       - Start Docker services"
	@echo "services-down     - Stop Docker services"

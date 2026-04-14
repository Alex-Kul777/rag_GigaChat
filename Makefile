# Makefile для управления тестами
.PHONY: test test-cov test-unit test-integration test-smoke clean help

help:
	@echo "Доступные команды:"
	@echo "  make test         - Запустить все тесты"
	@echo "  make test-cov     - Запустить тесты с coverage"
	@echo "  make test-unit    - Запустить только unit-тесты"
	@echo "  make test-smoke   - Запустить дымовые тесты"
	@echo "  make clean        - Очистить временные файлы"

test:
	pytest -v

test-cov:
	pytest --cov=. --cov-report=html --cov-report=term
	@echo "Coverage report: open htmlcov/index.html"

test-unit:
	pytest -m "unit" -v

test-smoke:
	pytest -m "smoke" -v

test-integration:
	pytest -m "integration" -v

clean:
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

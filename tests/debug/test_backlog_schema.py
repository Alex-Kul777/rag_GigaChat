"""
test_backlog_schema.py - Валидация YAML frontmatter всех BKL-*.md файлов
"""
import pytest
import yaml
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKLOG_DIR = PROJECT_ROOT / "backlog"


def get_backlog_files():
    """Получить все BKL-*.md файлы"""
    return list(BACKLOG_DIR.glob("BKL-*.md"))


def parse_frontmatter(file_path: Path) -> dict:
    """Распарсить YAML frontmatter из BKL файла"""
    with open(file_path, "r") as f:
        content = f.read()

    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter found in {file_path.name}")

    frontmatter = yaml.safe_load(match.group(1))
    return frontmatter


@pytest.mark.parametrize(
    "file_path",
    get_backlog_files(),
    ids=lambda p: p.name,
)
def test_backlog_schema(file_path: Path):
    """Валидировать схему BKL файла"""
    fm = parse_frontmatter(file_path)

    # Required fields
    assert fm.get("id"), f"{file_path.name}: 'id' field is required"
    assert fm.get("title"), f"{file_path.name}: 'title' field is required"
    assert fm.get("status"), f"{file_path.name}: 'status' field is required"
    assert fm.get("safety_checks"), f"{file_path.name}: 'safety_checks' must be non-empty list"

    # Validate format
    assert isinstance(fm.get("safety_checks"), list), "safety_checks must be a list"
    assert len(fm.get("safety_checks", [])) > 0, "safety_checks cannot be empty"

    # Validate enums
    allowed_status = {"open", "in-progress", "blocked", "done", "rejected"}
    assert fm.get("status") in allowed_status, f"Invalid status: {fm.get('status')}"

    allowed_priority = {"high", "medium", "low"}
    assert fm.get("priority") in allowed_priority, f"Invalid priority: {fm.get('priority')}"

    allowed_severity = {"critical", "major", "minor"}
    assert fm.get("severity") in allowed_severity, f"Invalid severity: {fm.get('severity')}"

    # Validate ID format
    assert re.match(r"^BKL-\d{3}$", fm.get("id", "")), f"Invalid ID format: {fm.get('id')}"


def test_backlog_files_exist():
    """Проверить что backlog директория и файлы существуют"""
    assert BACKLOG_DIR.exists(), f"Backlog directory not found: {BACKLOG_DIR}"

    files = get_backlog_files()
    assert len(files) > 0, "No BKL-*.md files found in backlog/"

    # Should have template and at least one example
    assert (BACKLOG_DIR / "template.md").exists(), "template.md not found"
    assert (BACKLOG_DIR / "BKL-000-example.md").exists(), "BKL-000-example.md not found"

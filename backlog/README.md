# Backlog Workflow Guide

## Overview
Backlog (`backlog/BKL-*.md`) is a centralized issue tracking system for the RAG debug cycle. Each entry links to log data via process mining signals.

## Backlog Entry Structure

Each file `backlog/BKL-NNN-<slug>.md` contains YAML frontmatter + Markdown body:

```yaml
---
id: BKL-001
title: "Short description (≤80 chars)"
priority: high              # high | medium | low
severity: major             # critical | major | minor
status: open                # open | in-progress | blocked | done | rejected
created: 2026-04-17
updated: 2026-04-17
affected_files:
  - src/rag_gigachat/data/pdf_loader.py
tags: [ocr, pdf, data-loading]
linked_logs: logs/session_20260417_1430.log#L120-L145
linked_events: logs/events_20260417_1430.csv
process_mining_evidence:
  variant: "V3"
  bottleneck_activity: "retrieval.vector_search"
  anomaly_type: "bimodal_distribution"
  affected_cases: 3
safety_checks:
  - pytest tests/unit/test_pdf_loader.py
rollback: "git reset --hard HEAD"
estimated_effort: 30min
---
```

## Priority & Severity Matrix

| Value | Priority (blocks work?) | Severity (incorrectness?) |
|-------|--------------------------|---------------------------|
| **high** | Critical path broken; data loss; security | App crash |
| **medium** | UX degradation; perf regression >20% | Function broken |
| **low** | Cosmetics; minor warnings | Inconvenient but OK |

**Orchestrator picks tasks**: `priority DESC, severity DESC, created ASC`

## Backlog Markdown Body

After the frontmatter, include:

### ## Проблема
Symptoms + log excerpts

### ## Гипотеза причины
(Optional) Why this is happening

### ## Предлагаемое исправление
Specific files/functions/diff sketch

### ## Критерии приёмки
- [ ] Log doesn't contain error X
- [ ] `pytest tests/unit/test_pdf_loader.py` — green
- [ ] (Optional) Manual check: load test_rotated.pdf

### ## Попытки
<!-- Orchestrator appends on each attempt -->

## Workflow States

```
open → in-progress → done ✅
             ↓
          blocked → (human review/rework)
```

- **open**: Ready for Orchestrator to pick
- **in-progress**: Orchestrator has claimed and created feature branch
- **blocked**: Pytest failed or critical decision needed
- **done**: Tests passed, commit created (awaiting git push)
- **rejected**: Won't fix or obsolete

## Safety Checks

### Pre-conditions
1. Worktree clean (`git status --porcelain`)
2. BKL entry valid, `status: open`
3. `safety_checks` = non-empty list of existing tests

### Post-conditions
1. All `safety_checks` passed
2. No regression in `pytest tests/unit/ -q`
3. Commit created: `fix: <title> (refs BKL-NNN)`

## Process Mining Evidence (optional)

Filled by Log Analyzer after `#analyze-logs`:
- **variant**: "V1", "V2", etc. — rare process variant
- **bottleneck_activity**: e.g. "retrieval.vector_search" — slow step
- **anomaly_type**: "rare_variant" | "bimodal_distribution" | "high_error_rate" | null
- **affected_cases**: Number of impacted traces

## Commands

```bash
# View backlog summary
python scripts/debug/backlog_index.py

# Create a new BKL entry (manual or from #analyze-logs)
cp backlog/template.md backlog/BKL-NNN-<slug>.md

# Apply fix (Orchestrator only)
#apply-fix BKL-NNN

# Change status without fixing
#reject BKL-NNN "Reason"
```

## Next Steps for New Maintainers

1. Check `backlog/INDEX.md` for open tasks
2. Pick highest-priority open task
3. Run `#apply-fix BKL-NNN`
4. Review & merge locally

---
name: Use uv for Python
description: User uses uv for Python package management, not pip/pip3
type: feedback
---

Use `uv run --with <package>` to run Python scripts with dependencies, not pip install.

**Why:** User has uv installed and prefers it over pip/pip3.
**How to apply:** When running Python scripts that need dependencies, use `uv run --with matplotlib,pandas,...` etc.

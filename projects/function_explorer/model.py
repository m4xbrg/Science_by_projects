# model.py — thin wrapper to expose FunctionExplorer for portfolio template
# Depends on function_explorer.py (kept at repo root for reuse across projects).

from function_explorer import FunctionExplorer  # re-export for template consistency

__all__ = ["FunctionExplorer"]

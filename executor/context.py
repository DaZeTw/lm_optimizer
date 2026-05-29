"""Runtime-only resources supplied by the executor.

Physical plan params are executable operator controls. Runtime dependencies
such as catalogs and caches live here instead of inside PhysicalNode.params.
"""

from __future__ import annotations

from dataclasses import dataclass

from catalog.catalog import SystemCatalog


@dataclass(frozen=True)
class ExecutionContext:
    catalog: SystemCatalog | None = None

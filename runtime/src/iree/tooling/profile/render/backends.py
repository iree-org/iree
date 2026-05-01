"""Renderer backend registry for iree-profile-render."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Callable, Sequence

from . import perfetto


@dataclasses.dataclass(frozen=True)
class RenderBackend:
    name: str
    description: str
    render: Callable[[list[dict[str, Any]], Path], Any]
    summary_fields: Callable[[Any], Sequence[tuple[str, Any]]]


BACKENDS = {
    perfetto.FORMAT_NAME: RenderBackend(
        name=perfetto.FORMAT_NAME,
        description=perfetto.FORMAT_DESCRIPTION,
        render=perfetto.render,
        summary_fields=perfetto.summary_fields,
    ),
}

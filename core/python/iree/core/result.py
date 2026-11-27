# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Results of importing and compiling a model, saved next to its artifacts."""

import dataclasses
import json
from pathlib import Path
from typing import Self, TypeVar

from dacite import Config, DaciteError, UnexpectedDataError, from_dict
from dacite.exceptions import DaciteFieldError

from ._paths import collect_paths
from .config import CompileConfig, ImportConfig, resolve_import_config_class

__all__ = [
    "CompileResult",
    "ImportResult",
]


_ResultT = TypeVar("_ResultT")

# Version of the metadata.json layout. Raise it when the layout changes in a
# way that older readers would misread.
_SCHEMA_VERSION = 1

# What a metadata.json file describes. Both kinds of result use that file name,
# so the file says which one it holds.
_IMPORT_RESULT = "import_result"
_COMPILE_RESULT = "compile_result"


class _PathSerializer:
    """JSON ``default`` that serializes each ``Path`` as its basename, after
    checking that it lives directly inside ``base_dir``.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._expected = base_dir.resolve()

    def __call__(self, obj: object) -> str:
        if isinstance(obj, Path):
            if obj.parent.resolve() != self._expected:
                raise ValueError(f"{obj} not in {self._base_dir}")
            if obj.name in ("", ".."):
                raise ValueError(f"{obj!r} does not name a file in {self._base_dir}")
            return obj.name
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _load_import_config(data: dict, config: Config) -> ImportConfig:
    """Build the import config with the class registered for its framework."""
    framework = data.get("framework", "")
    # A framework that is not a string has no class; the base class reports it.
    config_class = (
        resolve_import_config_class(framework)
        if isinstance(framework, str)
        else ImportConfig
    )
    try:
        return from_dict(data_class=config_class, data=data, config=config)
    except UnexpectedDataError as e:
        # The base class cannot hold importer-specific settings. They usually
        # belong to a class that is not registered yet.
        if config_class is ImportConfig:
            e.add_note(
                f"no import config class is registered for framework {framework!r}: "
                "import the module that defines it before loading"
            )
        raise


def _load_metadata_json(
    cls: type[_ResultT], metadata_path: Path, result_type: str
) -> _ResultT:
    """Load ``metadata_path``, a file of type ``result_type``, into ``cls``.

    Relative paths in the file are resolved against its directory.
    """
    base_dir = metadata_path.parent

    def resolve_path(path: object) -> Path:
        if not isinstance(path, (str, Path)):
            raise ValueError(f"expected a path, found {path!r}")
        # Joining keeps an absolute path as it is.
        return base_dir / path

    # JSON has no tuples: lists are cast to the tuple fields of a dataclass.
    # A key that matches no field is an error.
    config = Config(type_hooks={Path: resolve_path}, cast=[tuple], strict=True)
    try:
        # utf-8-sig also reads files that start with a byte order mark.
        data = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        data = dict(data)
        # Files written before these two keys existed have neither. Their
        # layout is that of version 1.
        file_type = data.pop("type", result_type)
        if file_type != result_type:
            raise ValueError(f"expected type {result_type!r}, found {file_type!r}")
        schema_version = data.pop("schema_version", 1)
        if type(schema_version) is not int or schema_version != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {schema_version!r}, "
                f"expected {_SCHEMA_VERSION}"
            )
        # Pre-build import_config with the correct subclass; dacite passes
        # pre-built instances through unchanged.
        if isinstance(data.get("import_config"), dict):
            try:
                data["import_config"] = _load_import_config(
                    data["import_config"], config
                )
            except DaciteFieldError as e:
                e.update_path("import_config")
                raise
        return from_dict(data_class=cls, data=data, config=config)
    except (json.JSONDecodeError, DaciteError, ValueError) as e:
        e.add_note(f"while loading {metadata_path}")
        raise


def _save_metadata_json(
    result: "ImportResult | CompileResult", result_type: str, output_dir: Path
) -> Path:
    """Write ``result`` of type ``result_type`` as metadata.json into ``output_dir``.

    Optional fields that are not set are left out of the file.
    """
    metadata_path = result.metadata_path(output_dir)
    data: dict[str, object] = {"type": result_type, "schema_version": _SCHEMA_VERSION}
    data.update(
        (key, value)
        for key, value in dataclasses.asdict(result).items()
        if value is not None
    )
    # Serialize before touching the file system, so that a path outside
    # output_dir is rejected without leaving a directory behind.
    text = json.dumps(data, indent=2, default=_PathSerializer(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(text, encoding="utf-8")
    return metadata_path


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImportResult:
    """
    Describes the artifacts produced by an importer.
    """

    imported_model: Path
    import_config: ImportConfig
    # The parameter archive that the imported model references but does not
    # contain, for importers that keep the weights out of the module.
    input_irpa: Path | None = None
    entry_function_name: str | None = None

    @property
    def output_dir(self) -> Path:
        """Return the directory containing the import output."""
        return self.imported_model.parent

    @classmethod
    def metadata_path(cls, output_dir: Path) -> Path:
        """Return the path to metadata.json inside *output_dir*."""
        return output_dir / "metadata.json"

    @staticmethod
    def is_metadata(path: Path) -> bool:
        """Check if a path points to a metadata.json file."""
        return path == ImportResult.metadata_path(path.parent)

    def save(self, output_dir: Path) -> Path:
        """Write metadata.json into output_dir. Returns path to metadata.json."""
        return _save_metadata_json(self, _IMPORT_RESULT, output_dir)

    @classmethod
    def load(cls, metadata_path: Path) -> Self:
        """Load from a local metadata.json file."""
        return _load_metadata_json(cls, metadata_path, _IMPORT_RESULT)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileResult:
    """Describes the artifacts produced by compiling a model.

    Artifact fields are paths to files on disk. ``load`` resolves relative ones
    against the directory of the metadata file.
    ``compile_config`` records the compile-time backend configuration and the
    runtime driver settings required to execute the artifact.
    """

    vmfb: Path
    compile_config: CompileConfig
    import_config: ImportConfig | None = None
    encoded_irpa: Path | None = None
    entry_function_name: str | None = None

    @property
    def vmfb_size(self) -> int:
        """Return the size of the vmfb in bytes."""
        return self.vmfb.stat().st_size

    @property
    def output_dir(self) -> Path:
        """Return the directory containing the compile output."""
        return self.vmfb.parent

    @classmethod
    def metadata_path(cls, output_dir: Path) -> Path:
        """Return the path to metadata.json inside *output_dir*."""
        return output_dir / "metadata.json"

    def get_model_artifacts(self) -> tuple[Path, ...]:
        """Serialized code and weights, excluding metadata and importer assets."""
        return (self.vmfb,) + (
            (self.encoded_irpa,) if self.encoded_irpa is not None else ()
        )

    def get_artifacts(self, *, include_metadata: bool = True) -> tuple[Path, ...]:
        """Return every file or directory needed to run this compile result."""
        paths = list(collect_paths(dataclasses.asdict(self)))
        if include_metadata:
            paths.insert(0, self.metadata_path(self.output_dir))
        return tuple(paths)

    def save(self, output_dir: Path) -> Path:
        """Write metadata.json into output_dir. Returns path to metadata.json."""
        return _save_metadata_json(self, _COMPILE_RESULT, output_dir)

    @staticmethod
    def is_metadata(path: Path) -> bool:
        """Check if a path points to a metadata.json file."""
        return path == CompileResult.metadata_path(path.parent)

    @classmethod
    def load(cls, metadata_path: Path) -> Self:
        """Load from a local metadata.json file."""
        return _load_metadata_json(cls, metadata_path, _COMPILE_RESULT)

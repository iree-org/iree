# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Configuration objects recorded with an imported or compiled model."""

import dataclasses
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Self, TypeVar

from dacite import Config, DaciteError, from_dict

from ._paths import collect_paths

__all__ = [
    "CompileConfig",
    "ConditionalRule",
    "ImportConfig",
    "register_import_config",
    "resolve_import_config_class",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ConditionalRule:
    """Flags that are only added when a condition holds.

    ``when`` maps the name of a condition to its arguments, for example
    ``{"arch_in": ["aarch64", "arm64"]}``. ``add`` lists the flags to add when
    the condition holds.
    """

    when: dict[str, list[str]]
    add: list[str] = dataclasses.field(default_factory=list)


def _copy_path_into(src: Path, dst_dir: Path) -> Path:
    """Copy a file or directory into ``dst_dir`` and return the copied path."""
    if src.name in ("", ".."):
        # Copying "." or ".." would copy a whole directory tree under no name.
        raise ValueError(f"cannot copy {src!r}: it does not name a file or directory")
    dst = dst_dir / src.name
    if src.resolve() == dst.resolve():
        return dst
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return dst


def _iter_direct_path_fields(obj: object) -> tuple[tuple[str, Path], ...]:
    """Return the ``(field_name, path)`` pairs for each direct ``Path`` field."""
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    fields: list[tuple[str, Path]] = []
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if isinstance(value, Path):
            fields.append((field.name, value))
    return tuple(fields)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ImportConfig:
    """
    Importer-side configuration recorded alongside an imported model.
    """

    framework: str

    def get_artifacts(self) -> tuple[Path, ...]:
        """Return importer-owned files or directories needed at runtime."""
        return collect_paths(dataclasses.asdict(self))

    def copy_to(self, dst_dir: Path) -> Self:
        """Copy the files that the ``Path`` fields refer to into ``dst_dir``.

        Only fields that are a ``Path`` themselves are copied, not paths inside
        lists or nested objects.

        Raises:
            ValueError: If two different files would get the same name in
                ``dst_dir``, ignoring case.
        """
        path_fields = _iter_direct_path_fields(self)
        first_with_name: dict[str, tuple[str, Path]] = {}
        for field_name, path in path_fields:
            # Names that differ only in case are one file on the file systems
            # that macOS and Windows use by default.
            other_field, other = first_with_name.setdefault(
                path.name.casefold(), (field_name, path)
            )
            if other.resolve() != path.resolve():
                raise ValueError(
                    f"cannot copy {other_field}={other} and {field_name}={path} "
                    f"into {dst_dir}: both would be named {path.name}, "
                    "ignoring case"
                )
        updates = {
            field_name: _copy_path_into(path, dst_dir)
            for field_name, path in path_fields
        }
        if not updates:
            return self
        return dataclasses.replace(self, **updates)


_ImportConfigT = TypeVar("_ImportConfigT", bound=ImportConfig)


# Maps ``ImportConfig.framework`` string to the concrete subclass.
_IMPORT_CONFIG_CLASSES: dict[str, type[ImportConfig]] = {}


def register_import_config(
    framework: str | list[str],
) -> Callable[[type[_ImportConfigT]], type[_ImportConfigT]]:
    """
    Decorator to register a concrete ImportConfig subclass for a given
    framework or list of frameworks.

    Raises:
        TypeError: If the decorated class is not an ImportConfig subclass.
        ValueError: If a framework is already registered to another class.
    """
    frameworks = [framework] if isinstance(framework, str) else framework

    def decorator(cls: type[_ImportConfigT]) -> type[_ImportConfigT]:
        if not (isinstance(cls, type) and issubclass(cls, ImportConfig)):
            raise TypeError(f"{cls!r} is not a subclass of ImportConfig")
        for fw in frameworks:
            registered = _IMPORT_CONFIG_CLASSES.get(fw)
            if registered is not None and registered is not cls:
                raise ValueError(
                    f"framework {fw!r} is already registered to "
                    f"{registered.__qualname__}"
                )
        for fw in frameworks:
            _IMPORT_CONFIG_CLASSES[fw] = cls
        return cls

    return decorator


def resolve_import_config_class(framework: str) -> type[ImportConfig]:
    """Return the concrete subclass for ``framework``, or the base class."""
    return _IMPORT_CONFIG_CLASSES.get(framework, ImportConfig)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompileConfig:
    """Configuration for compiling a model and for running the compiled artifact."""

    # Target device to use by iree-compile
    target_device: list[str]

    # Static flags to pass to iree-compile
    flags: list[str] = dataclasses.field(default_factory=list)

    # Flag templates: placeholders such as {target_triple} are filled in when
    # the model is compiled
    dynamic_flags: list[str] = dataclasses.field(default_factory=list)

    # Conditional flags are rules for adding flags only in certain cases
    conditional_flags: list[ConditionalRule] = dataclasses.field(default_factory=list)

    # Whether this compilation pipeline can compile IR whose parameters live in
    # an archive instead of the module. When false, the importer has to leave
    # the weights in the module.
    supports_external_parameters: bool = True

    # Runtime configuration: the drivers to create devices for, one device per
    # driver, in this order
    drivers: list[str]

    # Index into `drivers` of the lead device, which makes the allocations that
    # may be placed on any of the devices. Only used with more than one driver.
    # The runtime tools take it as `--device_lead_allocator`.
    device_lead_allocator: int = 0

    # Extra flags to be passed to the runtime
    runtime_flags: list[str] = dataclasses.field(default_factory=list)

    # Importer to use for a framework instead of its default one, by framework
    # name
    custom_importers: dict[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, file: str | Path) -> Self:
        """Load a configuration from a JSON file.

        A key that matches no field is an error.
        """
        file = Path(file)
        try:
            # utf-8-sig also reads files that start with a byte order mark.
            with file.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return from_dict(
                data_class=cls, data=data, config=Config(cast=[tuple], strict=True)
            )
        except (DaciteError, ValueError) as e:
            # Errors in the JSON and in its encoding are ValueErrors.
            e.add_note(f"while loading compile config from {file}")
            raise

    def to_dict(self) -> dict:
        """Return the configuration as JSON-serializable data."""
        return dataclasses.asdict(self)

    def save(self, file: str | Path) -> None:
        """Write the configuration to a JSON file that ``load`` reads back."""
        # Serialize first: a failure must not leave a truncated file behind.
        text = json.dumps(self.to_dict(), indent=4)
        Path(file).write_text(text, encoding="utf-8")

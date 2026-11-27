# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loading plugins and merging what they provide, built on pluggy.

A host, such as a compile driver, declares its hooks with ``hookspec`` and
creates a plugin manager for them with ``create_plugin_manager``. A plugin is a
module that implements hooks with ``hookimpl``. A package announces its plugin
with an entry point that names the module::

    [project.entry-points."<group>"]
    example = "example_plugin"

A hook through which plugins provide named values, such as the importer for a
framework, takes no arguments and returns a mapping from names to values. The
host merges the mappings of all plugins with ``merge_plugin_mappings``.

A host imports every plugin to learn what it provides, so a plugin module must
not import a heavy dependency, such as an ML framework, at its top level. It
provides factories that import the dependency when they are called::

    from iree.core.plugins import hookimpl

    def _create_importer(**kwargs):
        from example_plugin.importer import ExampleImporter  # Imports the framework.
        return ExampleImporter(**kwargs)

    @hookimpl
    def add_importer():
        return {"example": _create_importer}

To keep callers typed, a host annotates what the factories return with a base
class that is cheap to import.

All hosts share the markers and therefore one namespace of hooks: hook names
must be unique across hosts, and a plugin module implements the hooks of one
host.
"""

import importlib.metadata
import warnings
from collections.abc import Iterable, Mapping
from types import ModuleType
from typing import TypeVar

import pluggy

__all__ = [
    "create_plugin_manager",
    "hookimpl",
    "hookspec",
    "merge_plugin_mappings",
]

_PROJECT_NAME = "iree"

# Markers for the hooks of hosts and for their implementations in plugins. A
# plugin imports its marker from here, so that it need not import the host.
hookspec = pluggy.HookspecMarker(_PROJECT_NAME)
hookimpl = pluggy.HookimplMarker(_PROJECT_NAME)

_T = TypeVar("_T")


def create_plugin_manager(
    hookspecs: ModuleType | type, entry_point_group: str | None = None
) -> pluggy.PluginManager:
    """Create a plugin manager for the hooks that ``hookspecs`` declares.

    If ``entry_point_group`` is given, the plugins that installed packages
    declare in that group are loaded. A plugin that cannot be loaded is skipped
    with a warning and does not keep the others from loading. A hook that
    raises when it is called is not covered: it fails that call.
    """
    manager = pluggy.PluginManager(_PROJECT_NAME)
    manager.add_hookspecs(hookspecs)
    if entry_point_group is not None:
        _load_entry_point_plugins(manager, entry_point_group)
    return manager


def _distribution_name(distribution: importlib.metadata.Distribution) -> str:
    try:
        return f"distribution {distribution.metadata['Name']!r}"
    except Exception:
        return "a distribution without readable metadata"


def _load_entry_point_plugins(manager: pluggy.PluginManager, group: str) -> None:
    """Register the plugins in an entry point group, skipping broken ones.

    ``PluginManager.load_setuptools_entrypoints`` stops at the first
    distribution with unreadable metadata and at the first plugin that fails
    to import, which takes all other plugins down with it.
    """
    # The entry points that were loaded, or tried, by name.
    values: dict[str, str] = {}
    # A list: importing a plugin can change sys.path, which is being searched.
    for distribution in list(importlib.metadata.distributions()):
        try:
            entry_points = [
                entry_point
                for entry_point in distribution.entry_points
                if entry_point.group == group
            ]
        except Exception as e:
            warnings.warn(
                f"skipping {_distribution_name(distribution)}, whose entry points "
                f"cannot be read: {type(e).__name__}: {e}",
                stacklevel=3,
            )
            continue
        for entry_point in entry_points:
            plugin = f"plugin '{entry_point.name} = {entry_point.value}'"
            if entry_point.name in values:
                # The same distribution can be on sys.path more than once.
                # Another plugin under the same name is not the same thing.
                if values[entry_point.name] != entry_point.value:
                    warnings.warn(
                        f"skipping {plugin} of entry point group {group!r}: its name "
                        f"is taken by '{entry_point.name} = "
                        f"{values[entry_point.name]}'",
                        stacklevel=3,
                    )
                continue
            values[entry_point.name] = entry_point.value
            try:
                manager.register(entry_point.load(), name=entry_point.name)
            except Exception as e:
                # pluggy before 1.7 keeps a plugin whose hooks it rejects.
                if manager.get_plugin(entry_point.name) is not None:
                    manager.unregister(name=entry_point.name)
                warnings.warn(
                    f"skipping {plugin} of entry point group {group!r}: "
                    f"{type(e).__name__}: {e}",
                    stacklevel=3,
                )


def merge_plugin_mappings(
    kind: str, mappings: Iterable[Mapping[str, _T]]
) -> dict[str, _T]:
    """Merge the mappings that the plugins returned from a hook.

    ``kind`` names the values in the error message, for example ``"importer"``.

    Raises:
        ValueError: If more than one plugin provides a value for a name.
    """
    merged: dict[str, _T] = {}
    for mapping in mappings:
        for name, value in mapping.items():
            if name in merged:
                raise ValueError(f"more than one plugin provides the {kind} {name!r}")
            merged[name] = value
    return merged

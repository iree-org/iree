# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree.core.plugins."""

import importlib
import linecache
import sys
import tempfile
import textwrap
import unittest
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path

from iree.core.plugins import (
    create_plugin_manager,
    hookimpl,
    hookspec,
    merge_plugin_mappings,
)

GROUP = "iree.core.plugins_test"


class Hooks:
    """The hooks of a host that lets plugins provide widgets and gadgets."""

    @hookspec
    def add_widgets(self) -> dict[str, Callable[..., object]]:
        """Return the factories of the widgets that the plugin provides."""
        raise NotImplementedError

    @hookspec
    def add_gadgets(self) -> dict[str, Callable[..., object]]:
        """Return the factories of the gadgets that the plugin provides."""
        raise NotImplementedError


class NamedWidgets:
    """A plugin that is registered in code."""

    def __init__(self, *names: str):
        self._names = names

    @hookimpl
    def add_widgets(self) -> dict[str, Callable[..., object]]:
        return {name: dict for name in self._names}


class MergeTest(unittest.TestCase):
    def test_mappings_of_all_plugins_are_merged(self):
        manager = create_plugin_manager(Hooks)
        manager.register(NamedWidgets("a", "b"))
        manager.register(NamedWidgets("c"))

        widgets = merge_plugin_mappings("widget", manager.hook.add_widgets())

        self.assertEqual(widgets, {"a": dict, "b": dict, "c": dict})

    def test_no_plugin_gives_an_empty_mapping(self):
        manager = create_plugin_manager(Hooks)

        self.assertEqual(
            merge_plugin_mappings("widget", manager.hook.add_widgets()), {}
        )

    def test_a_name_from_two_plugins_is_rejected(self):
        manager = create_plugin_manager(Hooks)
        manager.register(NamedWidgets("a", "shared"))
        manager.register(NamedWidgets("shared"))

        with self.assertRaisesRegex(ValueError, "the widget 'shared'"):
            merge_plugin_mappings("widget", manager.hook.add_widgets())


class EntryPointTest(unittest.TestCase):
    """Plugins of installed packages, which are found through entry points."""

    PLUGIN = """
        from iree.core.plugins import hookimpl

        @hookimpl
        def add_widgets():
            return {"NAME": dict}
        """

    def setUp(self):
        # Module names of this test only, so that nothing else can import them.
        self.suffix = uuid.uuid4().hex
        self.addCleanup(self.forget_modules)
        self.site = self.add_site()

    def add_site(self) -> Path:
        """Create a directory that is searched for installed packages."""
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.addCleanup(sys.path.remove, temp_dir.name)
        sys.path.insert(0, temp_dir.name)
        importlib.invalidate_caches()
        return Path(temp_dir.name)

    def forget_modules(self):
        for name in [name for name in sys.modules if name.endswith(self.suffix)]:
            del sys.modules[name]
        importlib.invalidate_caches()

    def install(
        self,
        name: str,
        source: str,
        entry_points: str = "[{group}]\n{name} = {module}\n",
        site: Path | None = None,
    ) -> str:
        """Install a distribution with one plugin module; returns its name."""
        site = site or self.site
        module = f"{name}_{self.suffix}"
        (site / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
        dist_info = site / f"{module}-1.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {module}\nVersion: 1.0\n", encoding="utf-8"
        )
        (dist_info / "entry_points.txt").write_text(
            entry_points.format(group=GROUP, name=name, module=module),
            encoding="utf-8",
        )
        return module

    def widgets(self) -> dict[str, Callable[..., object]]:
        manager = create_plugin_manager(Hooks, GROUP)
        return merge_plugin_mappings("widget", manager.hook.add_widgets())

    def test_plugins_of_the_group_are_loaded(self):
        self.install("first", self.PLUGIN.replace("NAME", "one"))
        self.install("second", self.PLUGIN.replace("NAME", "two"))
        self.install(
            "other",
            self.PLUGIN.replace("NAME", "three"),
            "[another.group]\n{name} = {module}\n",
        )

        self.assertEqual(self.widgets(), {"one": dict, "two": dict})

    def test_names_can_be_computed_by_the_hook(self):
        self.install(
            "computed",
            """
            from iree.core.plugins import hookimpl

            @hookimpl
            def add_widgets():
                return {f"widget-{index}": dict for index in range(3)}
            """,
        )

        self.assertEqual(sorted(self.widgets()), ["widget-0", "widget-1", "widget-2"])

    def test_a_plugin_that_fails_to_import_is_skipped_with_a_warning(self):
        self.install("good", self.PLUGIN.replace("NAME", "one"))
        self.install("broken", "raise ImportError('no such framework')")

        with self.assertWarnsRegex(
            UserWarning, r"skipping plugin 'broken = .*no such framework"
        ) as warned:
            manager = create_plugin_manager(Hooks, GROUP)

        self.assertEqual(manager.hook.add_widgets(), [{"one": dict}])
        # The warning points at the code that created the plugin manager.
        self.assertEqual(warned.filename, __file__)
        self.assertIn(
            "create_plugin_manager(Hooks, GROUP)",
            linecache.getline(warned.filename, warned.lineno),
        )

    def test_a_plugin_that_does_not_match_the_hooks_is_skipped_as_a_whole(self):
        self.install(
            "mismatch",
            """
            from iree.core.plugins import hookimpl

            @hookimpl
            def add_widgets():
                return {"from-mismatch": dict}

            @hookimpl
            def add_gadgets(unexpected_argument):
                return {}
            """,
        )

        with self.assertWarnsRegex(UserWarning, "skipping plugin 'mismatch = "):
            manager = create_plugin_manager(Hooks, GROUP)

        # Its hook that does match is not used either.
        self.assertEqual(manager.hook.add_widgets(), [])
        self.assertIsNone(manager.get_plugin("mismatch"))

    def test_a_distribution_with_unreadable_entry_points_is_skipped(self):
        self.install("good", self.PLUGIN.replace("NAME", "one"))
        # A line without "=" is not an entry point.
        unreadable = self.install(
            "unreadable", "", "[console_scripts]\nnot an entry point\n"
        )

        with self.assertWarnsRegex(
            UserWarning, f"skipping distribution '{unreadable}', whose entry points"
        ) as warned:
            widgets = self.widgets()

        self.assertEqual(widgets, {"one": dict})
        self.assertEqual(warned.filename, __file__)

    def test_a_distribution_that_is_found_twice_is_loaded_once(self):
        for site in (self.site, self.add_site()):
            self.install("twice", self.PLUGIN.replace("NAME", "one"), site=site)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertEqual(self.widgets(), {"one": dict})

    def test_a_second_plugin_under_a_taken_name_is_skipped_with_a_warning(self):
        entry_points = "[{group}]\nplugin = {module}\n"
        self.install("first", self.PLUGIN.replace("NAME", "one"), entry_points)
        self.install("second", self.PLUGIN.replace("NAME", "two"), entry_points)

        with self.assertWarnsRegex(UserWarning, "its name is taken by 'plugin = "):
            widgets = self.widgets()

        # Which of the two is found first depends on the file system.
        self.assertEqual(len(widgets), 1)

    def test_a_factory_imports_its_dependency_only_when_it_is_called(self):
        # The stand-in for a framework that is expensive to import.
        framework = f"framework_{self.suffix}"
        (self.site / f"{framework}.py").write_text("VALUE = 42\n", encoding="utf-8")
        self.install(
            "lazy",
            f"""
            from iree.core.plugins import hookimpl

            def create_widget():
                import {framework}
                return {framework}.VALUE

            @hookimpl
            def add_widgets():
                return {{"lazy": create_widget}}
            """,
        )

        widgets = self.widgets()

        self.assertNotIn(framework, sys.modules)
        self.assertEqual(widgets["lazy"](), 42)
        self.assertIn(framework, sys.modules)


if __name__ == "__main__":
    unittest.main()

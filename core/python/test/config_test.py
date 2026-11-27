# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree.core.config."""

import dataclasses
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import Any

from dacite import DaciteError, UnexpectedDataError

from iree.core.config import (
    CompileConfig,
    ConditionalRule,
    ImportConfig,
    register_import_config,
    resolve_import_config_class,
)

DUMMY_FRAMEWORK = "dummy"


@register_import_config(DUMMY_FRAMEWORK)
@dataclasses.dataclass(frozen=True, kw_only=True)
class DummyImportConfig(ImportConfig):
    """Subclass that adds fields, including ``Path`` ones."""

    tokenizer: Path
    generation_config_dir: Path
    extra_setting: str = "default"


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp_path = Path(temp_dir.name)


class ImportConfigTest(TempDirTestCase):
    def test_get_artifacts_is_empty_without_path_fields(self):
        self.assertEqual(ImportConfig(framework="onnx").get_artifacts(), ())

    def test_get_artifacts_collects_path_fields(self):
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=self.tmp_path / "tokenizer",
            generation_config_dir=self.tmp_path / "generation_config",
        )
        self.assertEqual(
            config.get_artifacts(),
            (self.tmp_path / "tokenizer", self.tmp_path / "generation_config"),
        )

    def test_copy_to_returns_self_without_path_fields(self):
        config = ImportConfig(framework="onnx")
        self.assertIs(config.copy_to(self.tmp_path), config)

    def test_copy_to_copies_files_and_directories(self):
        src_dir = self.tmp_path / "src"
        dst_dir = self.tmp_path / "dst"
        (src_dir / "generation_config").mkdir(parents=True)
        (src_dir / "generation_config" / "config.json").write_text("{}")
        (src_dir / "tokenizer.json").write_text("tokens")
        dst_dir.mkdir()
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=src_dir / "tokenizer.json",
            generation_config_dir=src_dir / "generation_config",
            extra_setting="kept",
        )

        copied = config.copy_to(dst_dir)

        self.assertEqual(
            copied,
            DummyImportConfig(
                framework=DUMMY_FRAMEWORK,
                tokenizer=dst_dir / "tokenizer.json",
                generation_config_dir=dst_dir / "generation_config",
                extra_setting="kept",
            ),
        )
        self.assertEqual((dst_dir / "tokenizer.json").read_text(), "tokens")
        self.assertTrue((dst_dir / "generation_config" / "config.json").is_file())
        # The source config and its files are left as they were.
        self.assertEqual(config.tokenizer, src_dir / "tokenizer.json")
        self.assertTrue((src_dir / "tokenizer.json").is_file())

    def test_copy_to_keeps_files_that_are_already_in_place(self):
        (self.tmp_path / "generation_config").mkdir()
        (self.tmp_path / "tokenizer.json").write_text("tokens")
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=self.tmp_path / "tokenizer.json",
            generation_config_dir=self.tmp_path / "generation_config",
        )

        self.assertEqual(config.copy_to(self.tmp_path), config)
        self.assertEqual((self.tmp_path / "tokenizer.json").read_text(), "tokens")

    def test_copy_to_rejects_two_files_with_the_same_name(self):
        for directory in ("first", "second"):
            (self.tmp_path / directory).mkdir()
            (self.tmp_path / directory / "config.json").write_text(directory)
        dst_dir = self.tmp_path / "dst"
        dst_dir.mkdir()
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=self.tmp_path / "first" / "config.json",
            generation_config_dir=self.tmp_path / "second" / "config.json",
        )

        with self.assertRaisesRegex(ValueError, "both would be named config.json"):
            config.copy_to(dst_dir)

        # Nothing is copied when the names clash.
        self.assertEqual(list(dst_dir.iterdir()), [])

    def test_copy_to_rejects_names_that_differ_only_in_case(self):
        # On a file system that ignores case, the second copy would replace
        # the first.
        for directory, name in (
            ("first", "Tokenizer.json"),
            ("second", "tokenizer.json"),
        ):
            (self.tmp_path / directory).mkdir()
            (self.tmp_path / directory / name).write_text(directory)
        dst_dir = self.tmp_path / "dst"
        dst_dir.mkdir()
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=self.tmp_path / "first" / "Tokenizer.json",
            generation_config_dir=self.tmp_path / "second" / "tokenizer.json",
        )

        with self.assertRaisesRegex(ValueError, "ignoring case"):
            config.copy_to(dst_dir)

        self.assertEqual(list(dst_dir.iterdir()), [])

    def test_copy_to_accepts_the_same_file_in_two_fields(self):
        (self.tmp_path / "src").mkdir()
        (self.tmp_path / "src" / "shared.json").write_text("{}")
        dst_dir = self.tmp_path / "dst"
        dst_dir.mkdir()
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=self.tmp_path / "src" / "shared.json",
            generation_config_dir=self.tmp_path / "src" / "shared.json",
        )

        copied = config.copy_to(dst_dir)

        self.assertEqual(copied.tokenizer, dst_dir / "shared.json")
        self.assertEqual(copied.generation_config_dir, dst_dir / "shared.json")

    def test_copy_to_rejects_a_path_without_a_name(self):
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=Path(".."),
            generation_config_dir=self.tmp_path / "generation_config",
        )

        with self.assertRaisesRegex(ValueError, "does not name a file"):
            config.copy_to(self.tmp_path)


class ImportConfigRegistryTest(unittest.TestCase):
    def test_resolve_returns_registered_subclass(self):
        self.assertIs(resolve_import_config_class(DUMMY_FRAMEWORK), DummyImportConfig)

    def test_resolve_falls_back_to_base_class(self):
        self.assertIs(resolve_import_config_class("not-registered"), ImportConfig)

    def test_register_accepts_several_frameworks(self):
        # The registry lives as long as the process: use names of this run.
        first, second = f"first-{uuid.uuid4()}", f"second-{uuid.uuid4()}"

        @register_import_config([first, second])
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class SharedImportConfig(ImportConfig):
            pass

        self.assertIs(resolve_import_config_class(first), SharedImportConfig)
        self.assertIs(resolve_import_config_class(second), SharedImportConfig)

    def test_register_rejects_a_second_class_for_a_framework(self):
        with self.assertRaisesRegex(ValueError, "already registered to Dummy"):

            @register_import_config(["not-taken-yet", DUMMY_FRAMEWORK])
            @dataclasses.dataclass(frozen=True, kw_only=True)
            class OtherImportConfig(ImportConfig):
                pass

        # Nothing was registered, not even the framework that was still free.
        self.assertIs(resolve_import_config_class(DUMMY_FRAMEWORK), DummyImportConfig)
        self.assertIs(resolve_import_config_class("not-taken-yet"), ImportConfig)

    def test_register_accepts_the_same_class_again(self):
        self.assertIs(
            register_import_config(DUMMY_FRAMEWORK)(DummyImportConfig),
            DummyImportConfig,
        )

    def test_register_rejects_a_class_that_is_not_an_import_config(self):
        # Typed loosely, so that the mistake is only caught when it runs.
        not_a_config: Any = dict
        with self.assertRaisesRegex(TypeError, "not a subclass of ImportConfig"):
            register_import_config("not-a-config")(not_a_config)

        self.assertIs(resolve_import_config_class("not-a-config"), ImportConfig)


class CompileConfigTest(TempDirTestCase):
    def test_load_applies_defaults(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text(
            json.dumps({"target_device": ["local"], "drivers": ["local-task"]})
        )

        config = CompileConfig.load(config_path)

        self.assertEqual(
            config, CompileConfig(target_device=["local"], drivers=["local-task"])
        )
        self.assertEqual(config.flags, [])
        self.assertEqual(config.conditional_flags, [])
        self.assertTrue(config.supports_external_parameters)
        self.assertEqual(config.device_lead_allocator, 0)

    def test_load_builds_conditional_rules(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "target_device": ["local"],
                    "drivers": ["local-task"],
                    "conditional_flags": [
                        {"when": {"arch_in": ["aarch64"]}, "add": ["--flag"]},
                        {"when": {"arch_in": ["x86_64"]}},
                    ],
                }
            )
        )

        config = CompileConfig.load(config_path)

        self.assertEqual(
            config.conditional_flags,
            [
                ConditionalRule(when={"arch_in": ["aarch64"]}, add=["--flag"]),
                ConditionalRule(when={"arch_in": ["x86_64"]}, add=[]),
            ],
        )

    def test_save_and_load_roundtrip(self):
        config = CompileConfig(
            target_device=["local"],
            flags=["--static"],
            dynamic_flags=["--triple={target_triple}"],
            conditional_flags=[
                ConditionalRule(when={"arch_in": ["aarch64"]}, add=["--flag"])
            ],
            supports_external_parameters=False,
            drivers=["local-sync", "local-task"],
            device_lead_allocator=1,
            runtime_flags=["--runtime"],
            custom_importers={"onnx": "custom-onnx"},
        )
        config_path = self.tmp_path / "config.json"

        config.save(config_path)

        self.assertEqual(CompileConfig.load(config_path), config)
        self.assertEqual(json.loads(config_path.read_text()), config.to_dict())

    def test_load_accepts_a_string_path(self):
        config = CompileConfig(target_device=["local"], drivers=["local-task"])
        config_path = self.tmp_path / "config.json"
        config.save(str(config_path))

        self.assertEqual(CompileConfig.load(str(config_path)), config)

    def test_load_rejects_an_unknown_key(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "target_device": ["local"],
                    "drivers": ["local-task"],
                    "flag": ["--misspelled-key"],
                }
            )
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            CompileConfig.load(config_path)

        self.assertEqual(raised.exception.keys, {"flag"})
        self.assertIn(str(config_path), "\n".join(raised.exception.__notes__))

    def test_load_rejects_an_unknown_key_in_a_conditional_rule(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "target_device": ["local"],
                    "drivers": ["local-task"],
                    "conditional_flags": [
                        {"when": {}},
                        {"when": {}, "ad": ["--misspelled-key"]},
                    ],
                }
            )
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            CompileConfig.load(config_path)

        self.assertEqual(raised.exception.keys, {"ad"})

    def test_load_reads_a_file_with_a_byte_order_mark(self):
        config = CompileConfig(target_device=["local"], drivers=["local-task"])
        config_path = self.tmp_path / "config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8-sig")

        self.assertEqual(CompileConfig.load(config_path), config)

    def test_load_rejects_a_file_that_is_not_a_json_object(self):
        config_path = self.tmp_path / "config.json"
        for content in ("[]", "null", '"text"'):
            with self.subTest(content=content):
                config_path.write_text(content)

                with self.assertRaisesRegex(ValueError, "JSON object") as raised:
                    CompileConfig.load(config_path)

                self.assertIn(str(config_path), "\n".join(raised.exception.__notes__))

    def test_save_keeps_the_file_when_the_config_cannot_be_serialized(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text("kept")
        unserializable: Any = object()
        config = CompileConfig(
            target_device=["local"], drivers=["local-task"], flags=[unserializable]
        )

        with self.assertRaises(TypeError):
            config.save(config_path)

        self.assertEqual(config_path.read_text(), "kept")

    def test_load_names_the_file_when_a_field_is_missing(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text(json.dumps({"target_device": ["local"]}))

        with self.assertRaises(DaciteError) as raised:
            CompileConfig.load(config_path)

        self.assertIn("drivers", str(raised.exception))
        self.assertIn(str(config_path), "\n".join(raised.exception.__notes__))

    def test_load_names_the_file_when_json_is_invalid(self):
        config_path = self.tmp_path / "config.json"
        config_path.write_text("{not json")

        with self.assertRaises(json.JSONDecodeError) as raised:
            CompileConfig.load(config_path)

        self.assertIn(str(config_path), "\n".join(raised.exception.__notes__))


if __name__ == "__main__":
    unittest.main()

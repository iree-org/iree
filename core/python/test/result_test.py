# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for iree.core.result."""

import dataclasses
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dacite import DaciteError, UnexpectedDataError

from iree.core.config import CompileConfig, ImportConfig, register_import_config
from iree.core.result import CompileResult, ImportResult

# Unique to this file: all test files may share one process and one registry.
DUMMY_FRAMEWORK = "dummy-result"


@register_import_config(DUMMY_FRAMEWORK)
@dataclasses.dataclass(frozen=True, kw_only=True)
class DummyImportConfig(ImportConfig):
    """Subclass that adds fields, including ``Path`` ones."""

    tokenizer: Path
    generation_config_dir: Path
    extra_setting: str = "default"


@register_import_config("tuple-result")
@dataclasses.dataclass(frozen=True, kw_only=True)
class TupleImportConfig(ImportConfig):
    """Subclass with tuple fields, which JSON stores as lists."""

    files: tuple[Path, ...] = ()
    names: tuple[str, ...] = ()


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.tmp_path = Path(temp_dir.name)

    def write_metadata(self, output_dir: Path, data: object) -> Path:
        """Write ``data`` as metadata.json into ``output_dir``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(json.dumps(data), encoding="utf-8")
        return metadata_path

    def read_metadata(self, metadata_path: Path) -> dict:
        return json.loads(metadata_path.read_text(encoding="utf-8"))


def notes_of(exception: BaseException) -> str:
    return "\n".join(getattr(exception, "__notes__", []))


class ImportResultTest(TempDirTestCase):
    def test_save_writes_paths_as_basenames(self):
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=ImportConfig(framework="torch"),
            input_irpa=self.tmp_path / "weights.irpa",
        )

        metadata_path = result.save(self.tmp_path)

        self.assertEqual(metadata_path, ImportResult.metadata_path(self.tmp_path))
        self.assertEqual(
            self.read_metadata(metadata_path),
            {
                "type": "import_result",
                "schema_version": 1,
                "imported_model": "imported_model.mlirbc",
                "input_irpa": "weights.irpa",
                "import_config": {"framework": "torch"},
            },
        )

    def test_save_creates_the_output_directory(self):
        output_dir = self.tmp_path / "nested" / "output"
        result = ImportResult(
            imported_model=output_dir / "imported_model.mlirbc",
            import_config=ImportConfig(framework="torch"),
        )

        metadata_path = result.save(output_dir)

        self.assertTrue(metadata_path.is_file())

    def test_save_and_load_roundtrip(self):
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=ImportConfig(framework="torch"),
            input_irpa=self.tmp_path / "input.irpa",
            entry_function_name="forward",
        )

        loaded = ImportResult.load(result.save(self.tmp_path))

        self.assertEqual(loaded, result)

    def test_save_and_load_without_optional_fields(self):
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=ImportConfig(framework="onnx"),
        )

        metadata_path = result.save(self.tmp_path)

        # Optional fields that are not set are left out of the file.
        self.assertEqual(
            self.read_metadata(metadata_path),
            {
                "type": "import_result",
                "schema_version": 1,
                "imported_model": "imported_model.mlirbc",
                "import_config": {"framework": "onnx"},
            },
        )
        loaded = ImportResult.load(metadata_path)
        self.assertEqual(loaded, result)
        self.assertIsNone(loaded.input_irpa)
        self.assertIsNone(loaded.entry_function_name)

    def test_load_reads_null_as_unset(self):
        # save() leaves unset fields out, but null means the same.
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "imported_model": "imported_model.mlirbc",
                "input_irpa": None,
                "import_config": {"framework": "torch"},
                "entry_function_name": None,
            },
        )

        loaded = ImportResult.load(metadata_path)

        self.assertIsNone(loaded.input_irpa)
        self.assertIsNone(loaded.entry_function_name)

    def test_load_uses_registered_subclass(self):
        config = DummyImportConfig(
            framework=DUMMY_FRAMEWORK,
            tokenizer=self.tmp_path / "tokenizer.json",
            generation_config_dir=self.tmp_path / "generation_config",
            extra_setting="custom",
        )
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=config,
        )

        loaded = ImportResult.load(result.save(self.tmp_path))

        self.assertIsInstance(loaded.import_config, DummyImportConfig)
        self.assertEqual(loaded.import_config, config)

    def test_load_resolves_relative_paths_against_metadata_dir(self):
        output_dir = self.tmp_path / "output"
        metadata_path = self.write_metadata(
            output_dir,
            {
                "imported_model": "imported_model.mlirbc",
                "input_irpa": "weights.irpa",
                "import_config": {"framework": "torch"},
            },
        )

        loaded = ImportResult.load(metadata_path)

        self.assertEqual(loaded.imported_model, output_dir / "imported_model.mlirbc")
        self.assertEqual(loaded.input_irpa, output_dir / "weights.irpa")

    def test_load_resolves_subclass_paths_against_metadata_dir(self):
        output_dir = self.tmp_path / "output"
        metadata_path = self.write_metadata(
            output_dir,
            {
                "imported_model": "imported_model.mlirbc",
                "import_config": {
                    "framework": DUMMY_FRAMEWORK,
                    "tokenizer": "tokenizer.json",
                    "generation_config_dir": "generation_config",
                    "extra_setting": "x",
                },
            },
        )

        loaded = ImportResult.load(metadata_path)

        self.assertEqual(
            loaded.import_config,
            DummyImportConfig(
                framework=DUMMY_FRAMEWORK,
                tokenizer=output_dir / "tokenizer.json",
                generation_config_dir=output_dir / "generation_config",
                extra_setting="x",
            ),
        )

    def test_load_keeps_absolute_paths(self):
        # Absolute paths are never written by save(), but a hand-written file
        # may point outside its own directory.
        elsewhere = self.tmp_path / "elsewhere"
        output_dir = self.tmp_path / "output"
        metadata_path = self.write_metadata(
            output_dir,
            {
                "imported_model": str(elsewhere / "imported_model.mlirbc"),
                "input_irpa": "weights.irpa",
                "import_config": {"framework": "torch"},
            },
        )

        loaded = ImportResult.load(metadata_path)

        self.assertEqual(loaded.imported_model, elsewhere / "imported_model.mlirbc")
        self.assertEqual(loaded.input_irpa, output_dir / "weights.irpa")

    def test_save_rejects_imported_model_outside_output_dir(self):
        imported_model = self.tmp_path / "elsewhere" / "imported_model.mlirbc"
        result = ImportResult(
            imported_model=imported_model,
            import_config=ImportConfig(framework="torch"),
        )
        output_dir = self.tmp_path / "output"

        with self.assertRaisesRegex(ValueError, re.escape(str(imported_model))):
            result.save(output_dir)

        # No metadata is written that would lose the directory of the path.
        self.assertFalse(ImportResult.metadata_path(output_dir).exists())

    def test_save_rejects_subclass_path_field_outside_output_dir(self):
        # The check also covers the Path fields of import_config.
        tokenizer = self.tmp_path / "elsewhere" / "tokenizer.json"
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=DummyImportConfig(
                framework=DUMMY_FRAMEWORK,
                tokenizer=tokenizer,
                generation_config_dir=self.tmp_path / "generation_config",
            ),
        )

        with self.assertRaisesRegex(ValueError, re.escape(str(tokenizer))):
            result.save(self.tmp_path)

    def test_save_rejects_values_json_cannot_represent(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class UnserializableImportConfig(ImportConfig):
            setting: object

        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=UnserializableImportConfig(
                framework="unserializable", setting=object()
            ),
        )

        with self.assertRaisesRegex(TypeError, "is not JSON serializable"):
            result.save(self.tmp_path)

        self.assertFalse(ImportResult.metadata_path(self.tmp_path).exists())

    def test_load_names_the_file_when_a_field_is_missing(self):
        metadata_path = self.write_metadata(
            self.tmp_path, {"imported_model": "imported_model.mlirbc"}
        )

        with self.assertRaisesRegex(DaciteError, "import_config") as raised:
            ImportResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_rejects_a_file_that_is_not_an_object(self):
        metadata_path = self.write_metadata(self.tmp_path, ["imported_model.mlirbc"])

        with self.assertRaisesRegex(ValueError, "expected a JSON object") as raised:
            ImportResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_names_the_file_when_json_is_invalid(self):
        metadata_path = self.tmp_path / "metadata.json"
        metadata_path.write_text("{not json", encoding="utf-8")

        with self.assertRaises(json.JSONDecodeError) as raised:
            ImportResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_unregistered_framework_as_base_class(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "imported_model": "imported_model.mlirbc",
                "import_config": {"framework": "not-registered"},
            },
        )

        loaded = ImportResult.load(metadata_path)

        self.assertIs(type(loaded.import_config), ImportConfig)
        self.assertEqual(loaded.import_config, ImportConfig(framework="not-registered"))

    def test_load_rejects_settings_of_unregistered_framework(self):
        # The base class cannot hold the extra settings, so they must not be
        # dropped silently.
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "imported_model": "imported_model.mlirbc",
                "import_config": {
                    "framework": "not-registered",
                    "tokenizer": "tokenizer.json",
                },
            },
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            ImportResult.load(metadata_path)

        notes = notes_of(raised.exception)
        self.assertIn("'not-registered'", notes)
        self.assertIn(str(metadata_path), notes)

    def test_output_dir(self):
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=ImportConfig(framework="torch"),
        )
        self.assertEqual(result.output_dir, self.tmp_path)

    def test_metadata_path_is_inside_output_dir(self):
        self.assertEqual(
            ImportResult.metadata_path(self.tmp_path), self.tmp_path / "metadata.json"
        )

    def test_is_metadata(self):
        for name, expected in (
            ("metadata.json", True),
            ("imported_model.mlirbc", False),
            ("metadata.json.bak", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    ImportResult.is_metadata(self.tmp_path / "dir" / name), expected
                )


class CompileResultTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.compile_config = CompileConfig(
            target_device=["local"], drivers=["local-task"]
        )

    def full_result(self) -> CompileResult:
        """Return a result that sets every field."""
        return CompileResult(
            vmfb=self.tmp_path / "model.vmfb",
            compile_config=CompileConfig(
                target_device=["local"],
                flags=["--static"],
                drivers=["local-sync", "local-task"],
                device_lead_allocator=1,
                runtime_flags=["--runtime"],
            ),
            import_config=DummyImportConfig(
                framework=DUMMY_FRAMEWORK,
                tokenizer=self.tmp_path / "tokenizer",
                generation_config_dir=self.tmp_path / "generation_config",
                extra_setting="custom",
            ),
            encoded_irpa=self.tmp_path / "output.irpa",
            entry_function_name="forward",
        )

    def test_save_and_load_roundtrip_with_required_fields(self):
        result = CompileResult(
            vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
        )

        metadata_path = result.save(self.tmp_path)

        self.assertEqual(metadata_path, CompileResult.metadata_path(self.tmp_path))
        # Optional fields that are not set are left out of the file.
        self.assertEqual(
            self.read_metadata(metadata_path),
            {
                "type": "compile_result",
                "schema_version": 1,
                "vmfb": "model.vmfb",
                "compile_config": self.compile_config.to_dict(),
            },
        )
        self.assertEqual(CompileResult.load(metadata_path), result)

    def test_save_and_load_roundtrip_with_every_field(self):
        result = self.full_result()

        metadata_path = result.save(self.tmp_path)

        data = self.read_metadata(metadata_path)
        self.assertEqual(data["encoded_irpa"], "output.irpa")
        self.assertEqual(data["import_config"]["tokenizer"], "tokenizer")
        loaded = CompileResult.load(metadata_path)
        self.assertEqual(loaded, result)
        self.assertIsInstance(loaded.import_config, DummyImportConfig)

    def test_save_rejects_vmfb_outside_output_dir(self):
        vmfb = self.tmp_path / "elsewhere" / "model.vmfb"
        result = CompileResult(vmfb=vmfb, compile_config=self.compile_config)

        with self.assertRaisesRegex(ValueError, re.escape(str(vmfb))):
            result.save(self.tmp_path)

    def test_load_resolves_relative_paths_against_metadata_dir(self):
        output_dir = self.tmp_path / "output"
        metadata_path = self.write_metadata(
            output_dir,
            {
                "vmfb": "model.vmfb",
                "compile_config": self.compile_config.to_dict(),
                "encoded_irpa": "output.irpa",
            },
        )

        loaded = CompileResult.load(metadata_path)

        self.assertEqual(loaded.vmfb, output_dir / "model.vmfb")
        self.assertEqual(loaded.encoded_irpa, output_dir / "output.irpa")

    def test_load_keeps_absolute_paths(self):
        elsewhere = self.tmp_path / "elsewhere"
        metadata_path = self.write_metadata(
            self.tmp_path / "output",
            {
                "vmfb": str(elsewhere / "model.vmfb"),
                "compile_config": self.compile_config.to_dict(),
            },
        )

        loaded = CompileResult.load(metadata_path)

        self.assertEqual(loaded.vmfb, elsewhere / "model.vmfb")

    def test_tuple_fields_of_an_import_config_roundtrip(self):
        result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=TupleImportConfig(
                framework="tuple-result",
                files=(self.tmp_path / "a.bin", self.tmp_path / "b.bin"),
                names=("a", "b"),
            ),
        )

        self.assertEqual(ImportResult.load(result.save(self.tmp_path)), result)

    def test_failed_save_leaves_no_output_directory_behind(self):
        output_dir = self.tmp_path / "not-created"
        result = CompileResult(
            vmfb=self.tmp_path / "elsewhere" / "model.vmfb",
            compile_config=self.compile_config,
        )

        with self.assertRaises(ValueError):
            result.save(output_dir)

        self.assertFalse(output_dir.exists())

    def test_save_rejects_a_path_without_a_file_name(self):
        result = CompileResult(
            vmfb=self.tmp_path / "..", compile_config=self.compile_config
        )

        with self.assertRaisesRegex(ValueError, "does not name a file"):
            result.save(self.tmp_path)

    def test_load_rejects_a_path_that_is_not_a_string(self):
        for vmfb in (None, 5, ["model.vmfb"]):
            with self.subTest(vmfb=vmfb):
                metadata_path = self.write_metadata(
                    self.tmp_path,
                    {"vmfb": vmfb, "compile_config": self.compile_config.to_dict()},
                )

                with self.assertRaises((ValueError, DaciteError)) as raised:
                    CompileResult.load(metadata_path)

                self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_rejects_a_framework_that_is_not_a_string(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "imported_model": "imported_model.mlirbc",
                "import_config": {"framework": ["torch"]},
            },
        )

        with self.assertRaises(DaciteError) as raised:
            ImportResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_names_the_import_config_field_that_is_missing(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "imported_model": "imported_model.mlirbc",
                "import_config": {"framework": "dummy-result"},
            },
        )

        with self.assertRaisesRegex(DaciteError, r"import_config\.tokenizer"):
            ImportResult.load(metadata_path)

    def test_load_reads_a_file_with_a_byte_order_mark(self):
        result = CompileResult(
            vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
        )
        metadata_path = result.save(self.tmp_path)
        metadata_path.write_text(
            metadata_path.read_text(encoding="utf-8"), encoding="utf-8-sig"
        )

        self.assertEqual(CompileResult.load(metadata_path), result)

    def test_load_rejects_an_unknown_key(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "vmfb": "model.vmfb",
                "compile_config": self.compile_config.to_dict(),
                "zzz": 1,
            },
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            CompileResult.load(metadata_path)

        self.assertEqual(raised.exception.keys, {"zzz"})
        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_rejects_an_unknown_key_in_the_compile_config(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "vmfb": "model.vmfb",
                "compile_config": {**self.compile_config.to_dict(), "flag": []},
            },
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            CompileResult.load(metadata_path)

        self.assertEqual(raised.exception.keys, {"flag"})

    def test_load_rejects_an_unknown_key_in_a_registered_import_config(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "vmfb": "model.vmfb",
                "compile_config": self.compile_config.to_dict(),
                "import_config": {
                    "framework": "dummy-result",
                    "tokenizer": "tokenizer.json",
                    "generation_config_dir": "generation_config",
                    "extra": 1,
                },
            },
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            CompileResult.load(metadata_path)

        self.assertEqual(raised.exception.keys, {"extra"})
        # The framework has a class, so the hint about a missing one would mislead.
        self.assertNotIn("no import config class", notes_of(raised.exception))

    def test_load_rejects_a_file_of_the_other_type(self):
        import_result = ImportResult(
            imported_model=self.tmp_path / "imported_model.mlirbc",
            import_config=ImportConfig(framework="torch"),
        )
        metadata_path = import_result.save(self.tmp_path)

        with self.assertRaisesRegex(
            ValueError, "expected type 'compile_result', found 'import_result'"
        ) as raised:
            CompileResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_import_result_load_rejects_a_compile_result_file(self):
        compile_result = CompileResult(
            vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
        )
        metadata_path = compile_result.save(self.tmp_path)

        with self.assertRaisesRegex(
            ValueError, "expected type 'import_result', found 'compile_result'"
        ):
            ImportResult.load(metadata_path)

    def test_load_accepts_a_file_without_type_and_schema_version(self):
        # Files written before these two keys existed have neither.
        metadata_path = self.write_metadata(
            self.tmp_path,
            {"vmfb": "model.vmfb", "compile_config": self.compile_config.to_dict()},
        )

        loaded = CompileResult.load(metadata_path)

        self.assertEqual(
            loaded,
            CompileResult(
                vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
            ),
        )

    def test_load_reads_a_file_without_schema_version_as_version_1(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {"vmfb": "model.vmfb", "compile_config": self.compile_config.to_dict()},
        )

        # Stands in for a later version of the layout.
        with mock.patch("iree.core.result._SCHEMA_VERSION", 2):
            with self.assertRaisesRegex(ValueError, "unsupported schema_version 1,"):
                CompileResult.load(metadata_path)

    def test_load_rejects_other_schema_versions(self):
        for schema_version in (0, 2, "1", None):
            with self.subTest(schema_version=schema_version):
                metadata_path = self.write_metadata(
                    self.tmp_path,
                    {
                        "schema_version": schema_version,
                        "vmfb": "model.vmfb",
                        "compile_config": self.compile_config.to_dict(),
                    },
                )

                with self.assertRaisesRegex(ValueError, "schema_version") as raised:
                    CompileResult.load(metadata_path)

                self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_names_the_file_when_a_field_is_missing(self):
        metadata_path = self.write_metadata(self.tmp_path, {"vmfb": "model.vmfb"})

        with self.assertRaisesRegex(DaciteError, "compile_config") as raised:
            CompileResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_rejects_a_file_that_is_not_an_object(self):
        metadata_path = self.write_metadata(self.tmp_path, "model.vmfb")

        with self.assertRaisesRegex(ValueError, "expected a JSON object") as raised:
            CompileResult.load(metadata_path)

        self.assertIn(str(metadata_path), notes_of(raised.exception))

    def test_load_unregistered_framework_as_base_class(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "vmfb": "model.vmfb",
                "compile_config": self.compile_config.to_dict(),
                "import_config": {"framework": "not-registered"},
            },
        )

        loaded = CompileResult.load(metadata_path)

        self.assertIs(type(loaded.import_config), ImportConfig)
        self.assertEqual(loaded.import_config, ImportConfig(framework="not-registered"))

    def test_load_rejects_settings_of_unregistered_framework(self):
        metadata_path = self.write_metadata(
            self.tmp_path,
            {
                "vmfb": "model.vmfb",
                "compile_config": self.compile_config.to_dict(),
                "import_config": {
                    "framework": "not-registered",
                    "extra_setting": "lost",
                },
            },
        )

        with self.assertRaises(UnexpectedDataError) as raised:
            CompileResult.load(metadata_path)

        notes = notes_of(raised.exception)
        self.assertIn("'not-registered'", notes)
        self.assertIn(str(metadata_path), notes)

    def test_vmfb_size(self):
        vmfb = self.tmp_path / "model.vmfb"
        vmfb.write_bytes(b"\x00" * 16)
        result = CompileResult(vmfb=vmfb, compile_config=self.compile_config)
        self.assertEqual(result.vmfb_size, 16)

    def test_output_dir(self):
        result = CompileResult(
            vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
        )
        self.assertEqual(result.output_dir, self.tmp_path)

    def test_get_model_artifacts(self):
        result = self.full_result()
        self.assertEqual(
            result.get_model_artifacts(),
            (self.tmp_path / "model.vmfb", self.tmp_path / "output.irpa"),
        )

    def test_get_model_artifacts_without_irpa(self):
        result = CompileResult(
            vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
        )
        self.assertEqual(result.get_model_artifacts(), (self.tmp_path / "model.vmfb",))

    def test_get_artifacts_includes_importer_paths(self):
        # The tokenizer and generation config held by import_config are needed
        # at runtime too.
        result = self.full_result()
        artifacts = (
            self.tmp_path / "model.vmfb",
            self.tmp_path / "tokenizer",
            self.tmp_path / "generation_config",
            self.tmp_path / "output.irpa",
        )

        self.assertEqual(
            result.get_artifacts(), (self.tmp_path / "metadata.json",) + artifacts
        )
        self.assertEqual(result.get_artifacts(include_metadata=False), artifacts)

    def test_get_artifacts_with_required_fields(self):
        result = CompileResult(
            vmfb=self.tmp_path / "model.vmfb", compile_config=self.compile_config
        )
        self.assertEqual(
            result.get_artifacts(),
            (self.tmp_path / "metadata.json", self.tmp_path / "model.vmfb"),
        )
        self.assertEqual(
            result.get_artifacts(include_metadata=False),
            (self.tmp_path / "model.vmfb",),
        )

    def test_is_metadata(self):
        for name, expected in (
            ("metadata.json", True),
            ("model.vmfb", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    CompileResult.is_metadata(self.tmp_path / "dir" / name), expected
                )


if __name__ == "__main__":
    unittest.main()

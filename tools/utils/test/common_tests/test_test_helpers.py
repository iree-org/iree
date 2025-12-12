# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for test.test_helpers module."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Add project tools/utils to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from test.test_helpers import run_python_module


class TestRunPythonModule(unittest.TestCase):
    """Tests for run_python_module helper function."""

    def test_basic_invocation(self):
        """Test basic module invocation with arguments."""
        # Use a simple built-in module that's guaranteed to exist
        result = run_python_module(
            "json.tool",
            ["--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())

    def test_uses_current_python_executable(self):
        """Verify run_python_module uses sys.executable, not hardcoded 'python3'."""
        # Create a simple test module that prints sys.executable
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "test_module"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")
            (module_dir / "__main__.py").write_text(
                "import sys\nprint(sys.executable)\n"
            )

            # Add tmpdir to PYTHONPATH so we can import test_module
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            try:
                os.environ["PYTHONPATH"] = str(tmpdir)
                result = run_python_module(
                    "test_module",
                    [],
                    capture_output=True,
                    text=True,
                )
                # The subprocess should report the same executable we're using
                reported_executable = result.stdout.strip()
                self.assertEqual(reported_executable, sys.executable)
            finally:
                if original_pythonpath:
                    os.environ["PYTHONPATH"] = original_pythonpath
                elif "PYTHONPATH" in os.environ:
                    del os.environ["PYTHONPATH"]

    def test_pythonpath_setup(self):
        """Verify PYTHONPATH is set up correctly for tools/utils imports."""
        # Test that lit_tools module is importable (uses lit_tools.iree_lit_list)
        # Create a minimal test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write("// RUN: test\n")
            tmp_path = Path(tmp.name)

        try:
            result = run_python_module(
                "lit_tools.iree_lit_list",
                [str(tmp_path)],
                capture_output=True,
                text=True,
            )
            # If PYTHONPATH wasn't set correctly, the import would fail
            self.assertEqual(result.returncode, 0, f"stderr: {result.stderr}")
        finally:
            tmp_path.unlink()

    def test_kwargs_passed_through(self):
        """Verify subprocess kwargs are passed through correctly."""
        # Test with input parameter using json.tool which reads stdin
        result = run_python_module(
            "json.tool",
            [],
            input='{"test": "value"}',
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        # json.tool formats the input JSON
        self.assertIn('"test"', result.stdout)
        self.assertIn('"value"', result.stdout)

    def test_timeout_parameter(self):
        """Verify timeout parameter works."""
        # Create a module that sleeps
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "sleep_module"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")
            (module_dir / "__main__.py").write_text("import time\ntime.sleep(10)\n")

            # Add tmpdir to PYTHONPATH
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            try:
                os.environ["PYTHONPATH"] = str(tmpdir)
                # Test that timeout is respected
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_python_module(
                        "sleep_module",
                        [],
                        timeout=0.1,
                    )
            finally:
                if original_pythonpath:
                    os.environ["PYTHONPATH"] = original_pythonpath
                elif "PYTHONPATH" in os.environ:
                    del os.environ["PYTHONPATH"]

    def test_environment_preservation(self):
        """Verify existing environment variables are preserved."""
        # Create a module that prints an environment variable
        test_var = "TEST_RUN_PYTHON_MODULE_VAR"
        test_value = "test_value_12345"
        os.environ[test_var] = test_value

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                module_dir = Path(tmpdir) / "env_module"
                module_dir.mkdir()
                (module_dir / "__init__.py").write_text("")
                (module_dir / "__main__.py").write_text(
                    f"import os\nprint(os.environ.get('{test_var}', 'NOT_FOUND'))\n"
                )

                # Add tmpdir to PYTHONPATH
                original_pythonpath = os.environ.get("PYTHONPATH", "")
                try:
                    os.environ["PYTHONPATH"] = str(tmpdir)
                    result = run_python_module(
                        "env_module",
                        [],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0)
                    self.assertIn(test_value, result.stdout)
                finally:
                    if original_pythonpath:
                        os.environ["PYTHONPATH"] = original_pythonpath
                    elif "PYTHONPATH" in os.environ:
                        del os.environ["PYTHONPATH"]
        finally:
            # Clean up
            del os.environ[test_var]

    def test_pythonpath_preservation(self):
        """Verify existing PYTHONPATH is preserved and extended."""
        # Create a module that prints PYTHONPATH
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "path_module"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")
            (module_dir / "__main__.py").write_text(
                "import os\nprint(os.environ['PYTHONPATH'])\n"
            )

            # Set a custom PYTHONPATH
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            custom_path = "/custom/test/path"
            os.environ["PYTHONPATH"] = f"{str(tmpdir)}{os.pathsep}{custom_path}"

            try:
                result = run_python_module(
                    "path_module",
                    [],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0)
                # Both custom path and tools/utils should be in PYTHONPATH
                self.assertIn(custom_path, result.stdout)
                self.assertIn("tools/utils", result.stdout)
            finally:
                # Restore original
                if original_pythonpath:
                    os.environ["PYTHONPATH"] = original_pythonpath
                elif "PYTHONPATH" in os.environ:
                    del os.environ["PYTHONPATH"]

    def test_error_propagation(self):
        """Verify errors from subprocess are properly propagated."""
        # Create a module that exits with error code 42
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "error_module"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")
            (module_dir / "__main__.py").write_text("import sys\nsys.exit(42)\n")

            # Add tmpdir to PYTHONPATH
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            try:
                os.environ["PYTHONPATH"] = str(tmpdir)
                result = run_python_module(
                    "error_module",
                    [],
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 42)
            finally:
                if original_pythonpath:
                    os.environ["PYTHONPATH"] = original_pythonpath
                elif "PYTHONPATH" in os.environ:
                    del os.environ["PYTHONPATH"]

    def test_stderr_capture(self):
        """Verify stderr is captured when requested."""
        # Create a module that writes to stderr
        with tempfile.TemporaryDirectory() as tmpdir:
            module_dir = Path(tmpdir) / "stderr_module"
            module_dir.mkdir()
            (module_dir / "__init__.py").write_text("")
            (module_dir / "__main__.py").write_text(
                "import sys\nsys.stderr.write('ERROR MESSAGE\\n')\n"
            )

            # Add tmpdir to PYTHONPATH
            original_pythonpath = os.environ.get("PYTHONPATH", "")
            try:
                os.environ["PYTHONPATH"] = str(tmpdir)
                result = run_python_module(
                    "stderr_module",
                    [],
                    capture_output=True,
                    text=True,
                )
                self.assertIn("ERROR MESSAGE", result.stderr)
            finally:
                if original_pythonpath:
                    os.environ["PYTHONPATH"] = original_pythonpath
                elif "PYTHONPATH" in os.environ:
                    del os.environ["PYTHONPATH"]

    def test_lit_tools_module_invocation(self):
        """Test invoking a real lit_tools module (integration test)."""
        # Create a minimal test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as tmp:
            tmp.write(
                """// RUN: some-command
// CHECK: test
func.func @test() {
  return
}
"""
            )
            tmp_path = Path(tmp.name)

        try:
            # Use iree-lit-list which should work on any valid test file
            result = run_python_module(
                "lit_tools.iree_lit_list",
                [str(tmp_path)],
                capture_output=True,
                text=True,
            )
            # Should succeed and list the test
            self.assertEqual(
                result.returncode,
                0,
                f"stdout: {result.stdout}\nstderr: {result.stderr}",
            )
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()

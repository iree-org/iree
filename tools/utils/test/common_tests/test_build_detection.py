# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for common.build_detection module."""

import os

# Add project tools/utils to path for imports
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from common import build_detection


class TestDetectBuildDir(unittest.TestCase):
    """Tests for detect_build_dir function."""

    def setUp(self):
        """Create temporary directory structure for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        # Create worktree structure: /tmp/xxx/iree-loom/
        self.worktree = self.root / "iree-loom"
        self.worktree.mkdir()

        # Create build directory: /tmp/xxx/iree-loom-build/ with tools/ subdir.
        self.build_dir = self.root / "iree-loom-build"
        self.build_dir.mkdir()
        (self.build_dir / "tools").mkdir()

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        # Clear environment variable if set
        if "IREE_BUILD_DIR" in os.environ:
            del os.environ["IREE_BUILD_DIR"]

    def test_detect_from_worktree(self):
        """Test detection from worktree directory."""
        result = build_detection.detect_build_dir(cwd=self.worktree)
        self.assertEqual(result, self.build_dir)

    def test_environment_variable_override(self):
        """Test IREE_BUILD_DIR environment variable override."""
        custom_build = self.root / "custom-build"
        custom_build.mkdir()

        os.environ["IREE_BUILD_DIR"] = str(custom_build)
        result = build_detection.detect_build_dir(cwd=self.worktree)
        self.assertEqual(result, custom_build)

    def test_environment_variable_nonexistent(self):
        """Test error when IREE_BUILD_DIR points to nonexistent directory."""
        os.environ["IREE_BUILD_DIR"] = "/nonexistent/path"

        with self.assertRaises(FileNotFoundError) as cm:
            build_detection.detect_build_dir(cwd=self.worktree)

        self.assertIn("IREE_BUILD_DIR", str(cm.exception))
        self.assertIn("/nonexistent/path", str(cm.exception))

    def test_fallback_to_build_subdir(self):
        """Test fallback to ./build directory."""
        # Remove default build directory (need to remove tools/ first).
        (self.build_dir / "tools").rmdir()
        self.build_dir.rmdir()

        # Create ./build subdirectory with CMakeCache to mark it as valid build.
        build_subdir = self.worktree / "build"
        build_subdir.mkdir()
        (build_subdir / "CMakeCache.txt").touch()

        result = build_detection.detect_build_dir(cwd=self.worktree)
        self.assertEqual(result, build_subdir)

    def test_fallback_to_builds_dir(self):
        """Test fallback to ../builds/<name>/ directory."""
        # Remove default build directory (need to remove tools/ first).
        (self.build_dir / "tools").rmdir()
        self.build_dir.rmdir()

        # Create ../builds/iree-loom/ with tools/ to mark as valid.
        builds_dir = self.root / "builds" / "iree-loom"
        builds_dir.mkdir(parents=True)
        (builds_dir / "tools").mkdir()

        result = build_detection.detect_build_dir(cwd=self.worktree)
        self.assertEqual(result, builds_dir)

    def test_no_build_dir_found(self):
        """Test error when no build directory can be found."""
        # Remove build directory (need to remove tools/ first).
        (self.build_dir / "tools").rmdir()
        self.build_dir.rmdir()

        with self.assertRaises(FileNotFoundError) as cm:
            build_detection.detect_build_dir(cwd=self.worktree)

        error_msg = str(cm.exception)
        self.assertIn("Cannot find build directory", error_msg)
        self.assertIn("iree-loom-build", error_msg)
        self.assertIn("IREE_BUILD_DIR", error_msg)


class TestFindTool(unittest.TestCase):
    """Tests for find_tool function."""

    def setUp(self):
        """Create temporary build directory with tool locations."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.build_dir = Path(self.temp_dir.name)

        # Create tool directories
        self.tools_dir = self.build_dir / "tools"
        self.tools_dir.mkdir()

        self.llvm_bin = self.build_dir / "llvm-project" / "bin"
        self.llvm_bin.mkdir(parents=True)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_find_iree_tool(self):
        """Test finding IREE tool in tools/ directory."""
        iree_opt = self.tools_dir / "iree-opt"
        iree_opt.touch()

        result = build_detection.find_tool("iree-opt", build_dir=self.build_dir)
        self.assertEqual(result, iree_opt)

    def test_find_llvm_tool(self):
        """Test finding LLVM tool in llvm-project/bin/ directory."""
        filecheck = self.llvm_bin / "FileCheck"
        filecheck.touch()

        result = build_detection.find_tool("FileCheck", build_dir=self.build_dir)
        self.assertEqual(result, filecheck)

    def test_tool_not_found(self):
        """Test error when tool cannot be found."""
        with self.assertRaises(FileNotFoundError) as cm:
            build_detection.find_tool("nonexistent-tool", build_dir=self.build_dir)

        error_msg = str(cm.exception)
        self.assertIn("Cannot find tool 'nonexistent-tool'", error_msg)
        self.assertIn("tools/nonexistent-tool", error_msg)

    def test_auto_detect_build_dir(self):
        """Test auto-detection of build directory when not provided."""
        # This would require mocking detect_build_dir, skip for now
        pass


class TestIsDebugBuild(unittest.TestCase):
    """Tests for is_debug_build function."""

    def setUp(self):
        """Create temporary build directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.build_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_debug_build_assertions_on(self):
        """Test detection of debug build with assertions ON."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("LLVM_ENABLE_ASSERTIONS:BOOL=ON\n")

        result = build_detection.is_debug_build(build_dir=self.build_dir)
        self.assertTrue(result)

    def test_release_build_assertions_off(self):
        """Test detection of release build with assertions OFF."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("LLVM_ENABLE_ASSERTIONS:BOOL=OFF\n")

        result = build_detection.is_debug_build(build_dir=self.build_dir)
        self.assertFalse(result)

    def test_no_cmake_cache(self):
        """Test behavior when CMakeCache.txt doesn't exist."""
        result = build_detection.is_debug_build(build_dir=self.build_dir)
        self.assertFalse(result)

    def test_assertions_setting_not_found(self):
        """Test behavior when LLVM_ENABLE_ASSERTIONS not in cache."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("OTHER_SETTING:BOOL=ON\n")

        result = build_detection.is_debug_build(build_dir=self.build_dir)
        self.assertFalse(result)


class TestGetBuildType(unittest.TestCase):
    """Tests for get_build_type function."""

    def setUp(self):
        """Create temporary build directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.build_dir = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_debug_build_type(self):
        """Test detection of Debug build type."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("CMAKE_BUILD_TYPE:STRING=Debug\n")

        result = build_detection.get_build_type(build_dir=self.build_dir)
        self.assertEqual(result, "Debug")

    def test_release_build_type(self):
        """Test detection of Release build type."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\n")

        result = build_detection.get_build_type(build_dir=self.build_dir)
        self.assertEqual(result, "Release")

    def test_relwithdebinfo_build_type(self):
        """Test detection of RelWithDebInfo build type."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("CMAKE_BUILD_TYPE:STRING=RelWithDebInfo\n")

        result = build_detection.get_build_type(build_dir=self.build_dir)
        self.assertEqual(result, "RelWithDebInfo")

    def test_no_cmake_cache(self):
        """Test behavior when CMakeCache.txt doesn't exist."""
        result = build_detection.get_build_type(build_dir=self.build_dir)
        self.assertEqual(result, "Unknown")

    def test_build_type_not_found(self):
        """Test behavior when CMAKE_BUILD_TYPE not in cache."""
        cmake_cache = self.build_dir / "CMakeCache.txt"
        cmake_cache.write_text("OTHER_SETTING:STRING=Value\n")

        result = build_detection.get_build_type(build_dir=self.build_dir)
        self.assertEqual(result, "Unknown")


if __name__ == "__main__":
    unittest.main()

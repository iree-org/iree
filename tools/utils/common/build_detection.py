# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Auto-detect IREE build directories and provide helpful build suggestions.

Build directory search order:
1. IREE_BUILD_DIR environment variable (override)
2. ./build/ (in-tree build, standard CMake default)
3. ../<name>-build/ (sibling pattern: main → ../main-build/)
4. ../builds/<name>/ (builds directory: main → ../builds/main/)

When tools or build directories are not found, provides helpful error messages
suggesting how to build IREE.

This module auto-detects the appropriate build directory based on the current
working directory and provides utilities for finding build artifacts (tools, FileCheck).
"""

import os
from pathlib import Path


def find_repo_root(cwd: Path | None = None) -> Path | None:
    """Finds IREE repository root by walking up to find Bazel marker file.

    Args:
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        Path to repository root, or None if marker not found
    """
    if cwd is None:
        cwd = Path.cwd()

    # Walk up directory tree looking for Bazel marker files.
    # Check MODULE.bazel (bzlmod) first, then WORKSPACE (legacy).
    for parent in [cwd] + list(cwd.parents):
        if (parent / "MODULE.bazel").exists() or (parent / "WORKSPACE").exists():
            return parent
    return None


def _is_cmake_build_dir(path: Path) -> bool:
    """Checks if path is a valid CMake build directory.

    Validates by checking for CMakeCache.txt or tools/ subdirectory.
    This prevents false positives from empty directories or Python packages.

    Args:
        path: Directory to check

    Returns:
        True if path appears to be a CMake build directory
    """
    return (path / "CMakeCache.txt").exists() or (path / "tools").is_dir()


def detect_build_dir(cwd: Path | None = None) -> Path:
    """Detects build directory from current working directory.

    Search order:
    1. IREE_BUILD_DIR environment variable (override)
    2. ./build/ (in-tree build, standard CMake default)
    3. ../<name>-build/ (sibling pattern: main → ../main-build/)
    4. ../builds/<name>/ (builds directory: main → ../builds/main/)

    Args:
        cwd: Current working directory (defaults to os.getcwd())

    Returns:
        Path to build directory

    Raises:
        FileNotFoundError: If build directory cannot be found. Error message
            includes helpful suggestions for building IREE.

    Example:
        >>> # From /home/ben/src/iree/main/ with ./build/
        >>> detect_build_dir()
        PosixPath('/home/ben/src/iree/main/build')

        >>> # From /home/ben/src/iree/main/ with ../builds/main/
        >>> detect_build_dir()
        PosixPath('/home/ben/src/iree/builds/main')

        >>> # With IREE_BUILD_DIR override
        >>> os.environ['IREE_BUILD_DIR'] = '/custom/build'
        >>> detect_build_dir()
        PosixPath('/custom/build')
    """
    # Allow environment variable override
    if "IREE_BUILD_DIR" in os.environ:
        build_dir = Path(os.environ["IREE_BUILD_DIR"])
        if not build_dir.exists():
            raise FileNotFoundError(
                f"IREE_BUILD_DIR points to non-existent directory: {build_dir}"
            )
        return build_dir

    # Get current directory
    if cwd is None:
        cwd = Path.cwd()

    # Find repository root (works from any subdirectory).
    repo_root = find_repo_root(cwd)
    if repo_root is None:
        # Not in IREE repo - use cwd as fallback.
        repo_root = cwd

    parent_dir = repo_root.parent
    worktree_name = repo_root.name

    # Search order: <repo>/build/ -> ../<name>-build/ -> ../builds/<name>/
    # Priority to <repo>/build/ for standard CMake users.
    candidates = [
        repo_root / "build",  # In-tree: repo/build/
        parent_dir / f"{worktree_name}-build",  # Sibling: ../main-build/
        parent_dir / "builds" / worktree_name,  # Builds dir: ../builds/main/
    ]

    for candidate in candidates:
        if candidate.exists() and _is_cmake_build_dir(candidate):
            return candidate

    tried_paths = "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(candidates))
    raise FileNotFoundError(
        f"Cannot find build directory. Tried:\n{tried_paths}\n\n"
        f"Build IREE first:\n"
        f"  cmake -B {candidates[0]} -S {repo_root} && "
        f"cmake --build {candidates[0]} -j$(nproc)\n\n"
        f"Or set IREE_BUILD_DIR to specify a custom build location."
    )


def find_tool(tool_name: str, build_dir: Path | None = None) -> Path:
    """Finds IREE tool in build directory.

    Args:
        tool_name: Name of tool (e.g., 'iree-opt', 'FileCheck')
        build_dir: Build directory (auto-detected if not provided)

    Returns:
        Path to tool executable

    Raises:
        FileNotFoundError: If tool cannot be found

    Example:
        >>> find_tool('iree-opt')
        PosixPath('/home/ben/src/iree-loom-build/tools/iree-opt')

        >>> find_tool('FileCheck')
        PosixPath('/home/ben/src/iree-loom-build/llvm-project/bin/FileCheck')
    """
    if build_dir is None:
        build_dir = detect_build_dir()

    # Tool locations
    tool_paths = [
        build_dir / "tools" / tool_name,  # IREE tools
        build_dir / "bin" / tool_name,  # Alternative location
        build_dir / "llvm-project" / "bin" / tool_name,  # LLVM tools
    ]

    for tool_path in tool_paths:
        if tool_path.exists():
            return tool_path

    # Tool not found - provide helpful error message
    tried_paths = "\n".join(f"  - {p}" for p in tool_paths)
    raise FileNotFoundError(
        f"Cannot find tool '{tool_name}' in build directory {build_dir}.\n\n"
        f"Tried:\n{tried_paths}\n\n"
        f"Tool may not be built yet. Build IREE with:\n"
        f"  cmake --build {build_dir} --target {tool_name}\n\n"
        f"Or build all tools:\n"
        f"  cmake --build {build_dir} -j$(nproc)"
    )


def is_debug_build(build_dir: Path | None = None) -> bool:
    """Checks if build has LLVM assertions enabled (debug build).

    Args:
        build_dir: Build directory (auto-detected if not provided)

    Returns:
        True if LLVM_ENABLE_ASSERTIONS=ON, False otherwise

    Example:
        >>> is_debug_build()
        True  # If built with -DLLVM_ENABLE_ASSERTIONS=ON
    """
    if build_dir is None:
        build_dir = detect_build_dir()

    # Check CMakeCache.txt for LLVM_ENABLE_ASSERTIONS
    cmake_cache = build_dir / "CMakeCache.txt"
    if not cmake_cache.exists():
        # No CMakeCache, assume not debug
        return False

    with open(cmake_cache) as f:
        for line in f:
            if "LLVM_ENABLE_ASSERTIONS:BOOL=" in line:
                return "=ON" in line

    # Default to False if not found
    return False


def get_build_type(build_dir: Path | None = None) -> str:
    """Gets CMAKE_BUILD_TYPE from build directory.

    Args:
        build_dir: Build directory (auto-detected if not provided)

    Returns:
        Build type string (Debug, Release, RelWithDebInfo, MinSizeRel)
        Returns "Unknown" if cannot be determined

    Example:
        >>> get_build_type()
        'Debug'
    """
    if build_dir is None:
        build_dir = detect_build_dir()

    cmake_cache = build_dir / "CMakeCache.txt"
    if not cmake_cache.exists():
        return "Unknown"

    with open(cmake_cache) as f:
        for line in f:
            if "CMAKE_BUILD_TYPE:STRING=" in line:
                return line.split("=")[1].strip()

    return "Unknown"

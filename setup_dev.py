#!/usr/bin/env python
# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sets up the development tree with various kinds of defaults.

**DANGER WILL ROBINSON: This is experimental and is basically a serialization
of Stella's dev process. We aim to use this to standardize a uniform dev
process for out of the gate work by core team members, but the instructions
on the website remain the authoritative well-lit path.**

This script optimizes the case of a single development tree and CMake build
directory, taking care to use every trick we have to make the experience
uniform and with every development-appropriate performance trick we know
how to apply.

Usage:

  ./setup_dev.py

This will:

* Try to select an appropriate C compiler a linker, preferring clang and lld
  if available (controllable via --cc, --cxx, and --lld).
* Configure all parts of the project to use ccache if available (controllable
  via --no-ccache).
* Configure all projects to build in release mode with debug symbols and
  asserts enabled (configurable via --config).
* Configure all compiler components to build shared development libraries
  instead of static (controllable via --llvm-shared).
* Enable several optimizations for more efficient handling of debug data and
  static archives that have the side effect of making artifacts bound to the
  precise dev setup (controllable via --no-dev).
* Enables flags for Python.
* Write out a cheat_sheet.md file with more commands to do things.
* Print next steps to build IREE, run tests, etc.

You can also:

* Configure LLVM projects with a different build type via --llvm-config. This
  can be used, for example, to build the deps as Release (for vroom) but
  IREE more conservatively.
* Create separate build trees for LLVM/MLIR/LLD/Clang and set IREE up to depend
  on that vs doing full in-tree mondo builds (controllable via
  --llvm-external-build).

Things that still need work:

* Options to tweak santizers and set things up so that everything works with
  shared libraries, dynamic Pythons, etc.

"""

from typing import List, Optional

import argparse
from pathlib import Path

import os
import shutil
import subprocess
import sys

# Resolve the repo_root realpath, which allows us to symlink this file to
# the build directory and still have it work.
repo_root = Path(os.path.realpath(os.path.dirname(__file__)))

TOOLCHAIN_DEV_SETUP_CMAKE_TRAILER = """
# Thin archives makes static archives that only link to backing object files
# instead of embedding them. This makes them non-relocatable but is almost
# always the right thing outside of certain deployment/packaging scenarios.

execute_process(COMMAND ar -V OUTPUT_VARIABLE IREE_SETUP_DEV_AR_VERSION)
if ("${IREE_SETUP_DEV_AR_VERSION}" MATCHES "^GNU ar|LLVM")
  message(STATUS "Enabling thin archives (static libraries will not be relocatable)")
  set(CMAKE_AR ar)
  set(CMAKE_C_ARCHIVE_APPEND "<CMAKE_AR> qT <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> qT <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_C_ARCHIVE_CREATE "<CMAKE_AR> crT <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> crT <TARGET> <LINK_FLAGS> <OBJECTS>")
else()
  message(WARNING "Thin archives requested but not supported by ar")
endif()
unset(IREE_SETUP_DEV_AR_VERSION)

# Both LLVM and LLVM support flags to enable split-dwarf. Use those because
# it is tricky to enable correctly.
set(IREE_ENABLE_SPLIT_DWARF ON CACHE BOOL "Auto Configured" FORCE)
set(IREE_ENABLE_THIN_ARCHIVES ON CACHE BOOL "Auto Configured" FORCE)
set(LLVM_USE_SPLIT_DWARF ON CACHE BOOL "Auto Configured" FORCE)
"""

BUILD_CONFIG_TYPES = [
    "debug",
    "release",
    "release-asserts",
]


def build_config_to_options(build_config: str, project: str):
  """Returns a list of (name, value) tuples to activate a build config.

  Args:
    build_config: One of BUILD_CONFIG_TYPES.
    project: Either "iree" or "llvm".
  """
  if build_config == "debug":
    return [("CMAKE_BUILD_TYPE", "Debug")]
  elif build_config == "release":
    return [("CMAKE_BUILD_TYPE", "RelWithDebInfo")]
  elif build_config == "release-asserts":
    options = [("CMAKE_BUILD_TYPE", "RelWithDebInfo")]
    if project == "iree":
      options.append(("IREE_ENABLE_ASSERTIONS", "ON"))
    elif project == "llvm":
      options.append(("LLVM_ENABLE_ASSERTIONS", "ON"))
    else:
      raise ValueError(f"Unknown project: {project}")
    return options
  else:
    raise ValueError(f"Unknown build_config: {build_config}")


def build_config_to_args(build_config: str, project: str) -> List[str]:
  """Returns a list of '-D' command line args for a build config."""
  return [
      f"-D{t[0]}={t[1]}"
      for t in build_config_to_options(build_config, project)
  ]


class Config:
  """Encapsulates access to config settings that we persist."""

  toolchain_config_file_name = ".toolchain_config.cmake"

  def __init__(self, args):
    self.args = args

  @property
  def iree_build_dir(self) -> Path:
    return (repo_root / ".." / "iree-build").resolve()

  @property
  def deps_build_dir(self) -> Path:
    return (repo_root / ".." / "iree-deps").resolve()

  @property
  def llvm_build_config(self) -> str:
    return self.args.llvm_config or self.args.config

  def build_dir_suffix(self, build_type: str) -> str:
    return "" if build_type == "release-asserts" else f"-{build_type}"

  @property
  def llvm_build_dir(self) -> Path:
    return self.deps_build_dir / f"llvm-build{self.build_dir_suffix(self.llvm_build_config)}"

  @property
  def mlir_build_dir(self) -> Path:
    return self.deps_build_dir / f"mlir-build{self.build_dir_suffix(self.llvm_build_config)}"

  def has_toolchain_config(self) -> bool:
    return self.has_config_cmake(self.toolchain_config_file_name)

  def write_toolchain_config(self, dev_mode, cc_exe, cxx_exe, python_exe,
                             ccache_exe, enable_lld):
    """Writes .toolchain_config.cmake file."""
    lines = [
        f'set(Python3_EXECUTABLE "{python_exe}" CACHE PATH "Auto Configured Python Executable" FORCE)',
    ]
    # c compiler
    if cc_exe:
      lines.extend([
          f'if(NOT CMAKE_C_COMPILER STREQUAL "{cc_exe}")',
          f'  set(CMAKE_C_COMPILER "{cc_exe}" CACHE STRING "Auto Configured C compiler" FORCE)',
          f'endif()'
      ])
    if cxx_exe:
      lines.extend([
          f'if(NOT CMAKE_CXX_COMPILER STREQUAL "{cxx_exe}")',
          f'  set(CMAKE_CXX_COMPILER "{cxx_exe}" CACHE STRING "Auto Configured C++ compiler" FORCE)',
          f'endif()'
      ])

    # ccache
    if ccache_exe:
      lines.extend([
          f'set(CMAKE_C_COMPILER_LAUNCHER "{ccache_exe}" CACHE PATH "Auto Configured CCache Executable" FORCE)',
          f'set(CMAKE_CXX_COMPILER_LAUNCHER "{ccache_exe}" CACHE PATH "Auto Configured CCache Executable" FORCE)',
      ])

    # lld
    # It really sucks that there isn't a better way to do this. But see what
    # the various *_ENABLE_LLD tests on the CMake side do to understand why
    # this is hard.
    if enable_lld:
      lines.extend([
          f'set(IREE_ENABLE_LLD ON CACHE BOOL "Auto Configured lld" FORCE)',
          f'set(LLVM_USE_LINKER "lld" CACHE STRING "Auto Configured lld" FORCE)',
      ])

    contents = "\n".join(lines)

    # Additional CMake dev mode hacks to make it go vroom.
    if dev_mode:
      contents += TOOLCHAIN_DEV_SETUP_CMAKE_TRAILER

    self.write_config_cmake(self.toolchain_config_file_name, contents)

  def get_config_path(self, file_name: str) -> Path:
    return repo_root / file_name

  def has_config_cmake(self, file_name: str) -> bool:
    return self.get_config_path(file_name).exists()

  def write_config_cmake(self, file_name: str, contents: str):
    """Writes a named CMake config file (or does nothing on no change)."""
    config_path = self.get_config_path(file_name)
    try:
      with open(config_path, "rt") as f:
        existing_contents = f.read()
    except FileNotFoundError:
      existing_contents = None

    banner = "# AUTO-GENERATED BY setup_dev.py. DO NOT EDIT.\n"
    new_contents = banner + contents + "\n"
    if (existing_contents is None or (existing_contents.startswith(banner) and
                                      new_contents != existing_contents)):
      with open(config_path, "wt") as f:
        f.write(new_contents)


def parse_arguments():
  parser = argparse.ArgumentParser("IREE LLVM Setup")

  def add_boolean(parser, name, dest=None, **kwargs):
    if not dest:
      dest = name
    parser.add_argument(f"--{name}", dest=dest, action="store_true", **kwargs)
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")

  parser.add_argument(
      "--config",
      default="release-asserts",
      choices=BUILD_CONFIG_TYPES,
      help="Project wide configuration (defaults to release-asserts)")

  add_boolean(
      parser,
      "dev",
      default=True,
      help=
      "Enable dev mode, which makes things faster but artifacts will only work on this machine (default true)"
  )
  add_boolean(parser,
              "ccache",
              default=None,
              help="Enable or disable ccache (default auto)")
  add_boolean(parser,
              "lld",
              default=None,
              help="Enable or disable ccache (default auto)")
  default_cc = os.environ.get("CC")
  parser.add_argument(
      f"--cc",
      default=default_cc,
      help=
      f"Override the c compiler (or {default_cc if default_cc else 'auto-detect'} if omitted)"
  )
  default_cxx = os.environ.get("CXX")
  parser.add_argument(
      f"--cxx",
      default=default_cxx,
      help=
      f"Override the c++ compiler (or {default_cxx if default_cxx else 'auto-detect'} if omitted)"
  )

  add_boolean(
      parser,
      "llvm-external-build",
      dest="llvm_external_build",
      default=False,
      help=
      "Build LLVM/LLD/Clang/MLIR external to the main project (default True)")

  parser.add_argument(
      "--llvm-config",
      dest="llvm_config",
      default=None,
      choices=BUILD_CONFIG_TYPES,
      help="LLVM specific build type (defaults to project setting)")

  parser.add_argument(f"--llvm-no-configure",
                      action="store_false",
                      default=True,
                      dest="llvm_configure",
                      help="Skip configure stages")
  add_boolean(parser,
              "--llvm-shared",
              dest="llvm_shared",
              default=True,
              help="Enable LLVM shared library mode (default)")

  subparsers = parser.add_subparsers(dest="command", title='sub-command')
  parser_toolchain = subparsers.add_parser("toolchain",
                                           help="Regenerate toolchain config")
  args = parser.parse_args()
  return args


def log_verbose(msg: str):
  print(msg)


def die(msg: str):
  print(msg)
  sys.exit(1)


def find_command(cmd: str) -> Optional[str]:
  p = shutil.which(cmd)
  if p:
    log_verbose(f"- found {cmd}: {p}")
  else:
    log_verbose(f"- not found {cmd}")
  return p


def setup_toolchain(args, config: Config):
  # c/c++ compiler
  cc_exe = args.cc
  cxx_exe = args.cxx
  # Default to clang if omitted entirely.
  if cc_exe is None:
    cc_exe = find_command("clang")
  if cxx_exe is None:
    cxx_exe = find_command("clang++")

  # ccache
  ccache_path = None
  if args.ccache or args.ccache is None:
    ccache_path = find_command("ccache")
    if args.ccache and not ccache_path:
      die("ERROR: ccache enabled but not found. "
          "Ensure that 'ccache' is on the path")

  # lld
  enable_lld = False
  if args.lld or args.lld is None:
    enable_lld = bool(find_command("lld"))
    if args.lld and not enable_lld:
      die("ERROR: lld enabled but not found. Ensure that 'lld' is on your path")

  # dev mode
  if args.dev:
    log_verbose("- dev mode enabled (artifacts cannot be relocated)")
  config.write_toolchain_config(dev_mode=args.dev,
                                cc_exe=cc_exe,
                                cxx_exe=cxx_exe,
                                python_exe=sys.executable,
                                ccache_exe=ccache_path,
                                enable_lld=enable_lld)


def build_external_llvm(args, config: Config):
  # TODO: Make these configurable.
  llvm_main_src_dir = repo_root / "third_party" / "llvm-project"
  llvm_build_dir = config.llvm_build_dir
  mlir_build_dir = config.mlir_build_dir
  llvm_build_dir.mkdir(parents=True, exist_ok=True)
  mlir_build_dir.mkdir(parents=True, exist_ok=True)

  # Build type
  build_type_args = build_config_to_args(config.llvm_build_config, "llvm")

  # LLVM configure step.
  if args.llvm_configure:
    llvm_configure_cmake_args = [
        "cmake",
        "-GNinja",
        "-S",
        str(llvm_main_src_dir / "llvm"),
        "-B",
        str(llvm_build_dir),
        f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={config.get_config_path(config.toolchain_config_file_name)}",
        f"-DBUILD_SHARED_LIBS={'ON' if args.llvm_shared else 'OFF'}",
        "-DLLVM_ENABLE_PROJECTS=llvm;lld;clang",
        "-DLLVM_TARGETS_TO_BUILD=X86;ARM;AArch64;RISCV;NVPTX;WebAssembly;AMDGPU",
        "-DLLVM_BUILD_EXAMPLES=OFF",
        "-DLLVM_INSTALL_UTILS=ON",
    ]
    llvm_configure_cmake_args.extend(build_type_args)
    log_verbose(f"+ Configuring LLVM: {' '.join(llvm_configure_cmake_args)}")
    subprocess.check_call(llvm_configure_cmake_args)

  # LLVM build step.
  llvm_build_cmake_args = [
      "cmake",
      "--build",
      str(llvm_build_dir),
  ]
  log_verbose(f"+ Building LLVM: {' '.join(llvm_build_cmake_args)}")
  subprocess.check_call(llvm_build_cmake_args)

  # Locate LLVM cmake directories.
  def verify_cmake_dir(dir: Path):
    if not dir.is_dir:
      die(f"ERROR: Could not find build CMake dir: {dir}")

  llvm_cmake_dir = llvm_build_dir / "lib" / "cmake" / "llvm"
  verify_cmake_dir(llvm_cmake_dir)
  lld_cmake_dir = llvm_build_dir / "lib" / "cmake" / "lld"
  verify_cmake_dir(lld_cmake_dir)
  clang_cmake_dir = llvm_build_dir / "lib" / "cmake" / "clang"
  verify_cmake_dir(clang_cmake_dir)

  # MLIR configure step.
  if args.llvm_configure:
    mlir_configure_cmake_args = [
        "cmake",
        "-GNinja",
        "-S",
        str(llvm_main_src_dir / "mlir"),
        "-B",
        str(mlir_build_dir),
        f"-DCMAKE_PROJECT_TOP_LEVEL_INCLUDES={config.get_config_path(config.toolchain_config_file_name)}",
        f"-DBUILD_SHARED_LIBS={'ON' if args.llvm_shared else 'OFF'}",
        f"-DLLVM_DIR={llvm_cmake_dir}",
        "-DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF",
        "-DLLVM_BUILD_TOOLS=ON",
        "-DMLIR_ENABLE_BINDINGS_PYTHON=ON",
    ]
    mlir_configure_cmake_args.extend(build_type_args)
    log_verbose(f"+ Configuring MLIR: {' '.join(mlir_configure_cmake_args)}")
    subprocess.check_call(mlir_configure_cmake_args)

  # MLIR build step.
  mlir_build_cmake_args = [
      "cmake",
      "--build",
      str(mlir_build_dir),
  ]
  log_verbose(f"+ Building MLIR: {' '.join(mlir_build_cmake_args)}")
  subprocess.check_call(mlir_build_cmake_args)

  # Locate MLIR cmake directories.
  mlir_cmake_dir = mlir_build_dir / "lib" / "cmake" / "mlir"
  verify_cmake_dir(mlir_cmake_dir)

  # Write llvm dep config.
  config_lines = [
      f'set(LLVM_DIR "{llvm_cmake_dir}" CACHE STRING "Auto Configured CMake dir" FORCE)',
      f'set(LLD_DIR "{lld_cmake_dir}" CACHE STRING "Auto Configured CMake dir" FORCE)',
      f'set(Clang_DIR "{clang_cmake_dir}" CACHE STRING "Auto Configured CMake dir" FORCE)',
      f'set(MLIR_DIR "{mlir_cmake_dir}" CACHE STRING "Auto Configured CMake dir" FORCE)',
      f'set(IREE_BUILD_BUNDLED_LLVM OFF CACHE BOOL "Auto Configured setting" FORCE)',
      f'set(IREE_COMPILER_BUILD_SHARED_LIBS {"ON" if args.llvm_shared else "OFF"} CACHE BOOL "Auto Configured setting" FORCE)',
  ]
  config.write_config_cmake(".llvm_dep_config.cmake", "\n".join(config_lines))


def build_internal_llvm(args, config: Config):
  # Write llvm dep config.
  config_lines = [
      f'set(IREE_BUILD_BUNDLED_LLVM ON CACHE BOOL "Auto Configured setting" FORCE)',
      f'set(IREE_COMPILER_BUILD_SHARED_LIBS {"ON" if args.llvm_shared else "OFF"} CACHE BOOL "Auto Configured setting" FORCE)',
  ]
  config.write_config_cmake(".llvm_dep_config.cmake", "\n".join(config_lines))


def main(args):
  config = Config(args)
  log_verbose(f"- generating {config.toolchain_config_file_name}")
  setup_toolchain(args, config)

  # Build LLVM and MLIR.
  if args.llvm_external_build:
    build_external_llvm(args, config)
  else:
    build_internal_llvm(args, config)

  next_steps = f"""# Initial CMake configure.
cmake -GNinja -S{repo_root} -B{config.iree_build_dir} {' '.join(build_config_to_args(args.config, "iree"))} -DIREE_BUILD_PYTHON_BINDINGS=ON
# Build project.
cmake --build {config.iree_build_dir}
# Build project and run tests.
cmake --build {config.iree_build_dir} --target iree-run-tests
"""

  # Print a cheat-sheet.
  cheatsheet = f"""# IREE Development Cheat Sheet

## Initial build

```
{next_steps}
```

## Rebuild LLVM/LLD/Clang/MLIR

```
# Rebuild LLVM+LLD+Clang
cmake --build {config.llvm_build_dir}
# Rebuild MLIR
cmake --build {config.mlir_build_dir}
```
"""
  cheatsheet_path = repo_root / "dev_cheatsheet.md"
  with open(repo_root / "dev_cheatsheet.md", "wt") as f:
    f.write(cheatsheet)
  log_verbose(f"wrote cheatsheet to {cheatsheet_path}")

  # Some next step commands.
  print()
  print("Next steps:")
  print("-----------")
  print(next_steps)


if __name__ == "__main__":
  main(parse_arguments())
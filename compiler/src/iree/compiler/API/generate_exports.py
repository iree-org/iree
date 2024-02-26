#!/usr/bin/env python
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Scans sources and generates export files.

Since we are only exporting a controlled C-API which is intended to be
standalone and present minimal chance for conflicts, we take an explicit
approach to generate lists of export symbols. Further, in order to ease
the integration with the build system, we assemble exported shared libraries
from normal static libraries that have been annotated properly at the source
level with visibility and/or dllexport attributes (depending on platform).

While it is possible to forgo this explicit control and twist the build
system to force link leaf objects, exporting annotated symbols, in practice,
this is quite hard to manage with a level of precision that these things need.
Also, it runs quite counter to how build systems reason about library
dependencies and is very invasive.

Instead, once we have our list of symbols, we generate:

* api_exports.ld : A GNU-style linker script for setting up exports.
* api_exports.lst : A MacOS-style file suitable to pass to
  --exported_symbols_list
* api_exports.def : A Windows def file.
* api_exports.c : Source file that triggers extern resolution for the symbols
  we want (this is more portable/flexible vs --undefined at the expense of
  feeling like a hack - spoilers: building shared libraries is a hack).

These are files generated and checked into the repo. In this way, a human is
in the loop on API symbol export and the API surface area becomes clearly
documented in the source control system (vs automagically/opaquely at a low
level of the build).
"""

from pathlib import Path
import re
from typing import List

LOCAL_HEADER_FILES = [
    "../../../../bindings/c/iree/compiler/embedding_api.h",
    "../../../../bindings/c/iree/compiler/tool_entry_points_api.h",
    "../../../../bindings/c/iree/compiler/mlir_interop.h",
]

MLIR_C_HEADER_FILES = [
    "AffineExpr.h",
    "AffineMap.h",
    "BuiltinAttributes.h",
    "BuiltinTypes.h",
    "Debug.h",
    "Diagnostics.h",
    "IntegerSet.h",
    "Interfaces.h",
    "IR.h",
    "Pass.h",
    "Support.h",
    "Transforms.h",
    "Dialect/Linalg.h",
    "Dialect/Transform.h",
    "Dialect/Transform/Interpreter.h",
    "Dialect/PDL.h",
]

IREE_DIALECTS_HEADER_FILES = [
    "Dialects.h",
]

EXPLICIT_EXPORTS = [
    # MLIR registration functions that are part of generated code.
    "mlirRegisterLinalgPasses",
    "mlirGetDialectHandle__iree_input__",
    "mlirGetDialectHandle__transform__",
]

# Matches statements that start with a well-known function declaration macro.
# The group 'decl' contains the statement after the macro.
MACRO_STATEMENT_PATTERN = re.compile(
    r"\n(MLIR_CAPI_EXPORTED|IREE_EMBED_EXPORTED)\s+(?P<decl>[^\;]+);",
    re.MULTILINE | re.DOTALL,
)

# Given a statement suspected to be a function declaration, extract the
# function symbol.
FUNC_DECL_SYMBOL_PATTERN = re.compile(r"(?P<symbol>\w+)\(", re.MULTILINE | re.DOTALL)


def main(repo_root: Path, api_root: Path):
    export_symbols = list(EXPLICIT_EXPORTS)
    # Collect symbols from local header files.
    for local_name in LOCAL_HEADER_FILES:
        export_symbols.extend(collect_header_exports(api_root / local_name))

    # Collect symbols from iree-dialects header files.
    for local_name in IREE_DIALECTS_HEADER_FILES:
        export_symbols.extend(
            collect_header_exports(
                repo_root
                / "llvm-external-projects/iree-dialects/include/iree-dialects-c"
                / local_name
            )
        )

    # Collect symbols from mlir-c header files.
    mlir_c_dir = repo_root / "third_party/llvm-project/mlir/include/mlir-c"
    for local_name in MLIR_C_HEADER_FILES:
        header_file = mlir_c_dir / local_name
        if not header_file.exists():
            raise RuntimeError(
                f"Expected MLIR-C header file does not exist: {header_file}"
            )
        export_symbols.extend(collect_header_exports(header_file))

    # Generate.
    export_symbols.sort()
    generate_macos_symbol_list(export_symbols, api_root / "api_exports.macos.lst")
    generate_linker_script(export_symbols, api_root / "api_exports.ld")
    generate_def_file(export_symbols, api_root / "api_exports.def")
    generate_force_extern(export_symbols, api_root / "api_exports.c")


def collect_header_exports(header_file: Path):
    with open(header_file, "r") as f:
        contents = f.read()

    symbols = []
    for m in re.finditer(MACRO_STATEMENT_PATTERN, contents):
        decl = m.group("decl")
        decl_m = re.search(FUNC_DECL_SYMBOL_PATTERN, decl)
        if decl_m:
            symbol = decl_m.group("symbol")
            symbols.append(symbol)
    return symbols


def generate_macos_symbol_list(symbols: List[str], file: Path):
    with open(file, "wt") as f:
        f.write("# Generated by generate_exports.py: Do not edit.\n")
        for symbol in symbols:
            # Note that cdecl symbols on MacOS are prefixed with "_", same as
            # we all did in the 80s but (thankfully) allowing longer than 8 character
            # names.
            f.write(f"_{symbol}\n")


def generate_linker_script(symbols: List[str], file: Path):
    with open(file, "wt") as f:
        f.write("# Generated by generate_exports.py: Do not edit.\n")
        f.write("VER_0 {\n")
        f.write("  global:\n")
        for symbol in symbols:
            f.write(f"    {symbol};\n")
        f.write("  local:\n")
        f.write("    *;\n")
        f.write("};\n")


def generate_def_file(symbols: List[str], file: Path):
    with open(file, "wt") as f:
        f.write("; Generated by generate_exports.py: Do not edit.\n")
        f.write("EXPORTS\n")
        for symbol in symbols:
            f.write(f"  {symbol}\n")


def generate_force_extern(symbols: List[str], file: Path):
    with open(file, "wt") as f:
        f.write("// Copyright 2022 The IREE Authors\n")
        f.write("//\n")
        f.write("// Licensed under the Apache License v2.0 with LLVM Exceptions.\n")
        f.write("// See https://llvm.org/LICENSE.txt for license information.\n")
        f.write("// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n")
        f.write("\n")
        f.write("// Generated by generate_exports.py: Do not edit.\n")
        f.write("\n")
        f.write("#include <stdint.h>\n")
        f.write("\n")
        for symbol in symbols:
            f.write(f"extern void {symbol}();\n")
        f.write("\n")
        f.write("uintptr_t __iree_compiler_hidden_force_extern() {\n")
        f.write("  uintptr_t x = 0;\n")
        for symbol in symbols:
            f.write(f"  x += (uintptr_t)&{symbol};\n")
        f.write("  return x;\n")
        f.write("}\n")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    repo_root = script_dir
    while True:
        # Key off of "AUTHORS" file
        if (repo_root / "AUTHORS").exists():
            break
        repo_root = repo_root.parent
        if not repo_root:
            raise RuntimeError(f"Could not find root of repo from {script_dir}")

    main(repo_root=repo_root, api_root=script_dir)

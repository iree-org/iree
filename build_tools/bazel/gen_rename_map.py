#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates an llvm-objcopy --redefine-syms map for IREE ABI renaming.

Renaming lets the compiler library share a process with another LLVM/MLIR copy,
TensorFlow's say. Plugins rename to match, so shared symbols resolve against the
library and MLIR TypeIDs coalesce across the dlopen boundary.

The decision is made on the demangled name, which must mention `llvm::` or
`mlir::` as a namespace. Matching the mangled string instead would be unsound:
`_Z9good4mlirv` (`good4mlir()`) contains `4mlir` inside a longer source-name.
Symbols merely naming an llvm/mlir type in their signature are renamed too, or
plugin functions would disagree with their call sites.

The inserted component is length-prefixed so the result still demangles:

  _ZN4mlir3fooEv        -> _ZN6IREE184mlir3fooEv    IREE18::mlir::foo()
  _ZTVN4mlir6WalkerE    -> _ZTVN6IREE184mlir6WalkerE  vtable for IREE18::...
  _Z3fooN4mlir5ValueE   -> _Z6IREE183fooN4mlir5ValueE IREE18(foo, mlir::Value)

The transformation is a pure function of the mangled name, so the library and
every renamed plugin archive arrive at the same new name independently.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

ITANIUM_MANGLED_PREFIXES = ("__Z", "_Z")

# Special encodings that may precede the nested-name: vtable, VTT, typeinfo,
# typeinfo name, guard variable, reference temporary, thread-local init and
# wrapper, and `Z` for function-local scope (these chain, e.g. `GVZN...` is
# the guard variable of a local static inside a namespaced function). The
# component is inserted after the `N` so the result stays demanglable.
# Anything else (thunks and other rare forms) falls back to insertion right
# after the mangling prefix: the result may not demangle, but the rename
# stays consistent everywhere.
_MARKED_NESTED_RE = re.compile(r"^(?:GV|GR|TV|TT|TI|TS|TH|TW|Z)*N")

# Namespace qualifier of llvm:: or mlir:: anywhere in the demangled name,
# including argument and template positions. The look-behind rejects
# namespaces that merely end in llvm/mlir (e.g. `myllvm::`).
_NEEDS_RENAME_RE = re.compile(r"(?<![A-Za-z0-9_])(llvm|mlir)::")


def _itanium_component(symbol_prefix: str) -> str:
    """Returns the length-prefixed Itanium source-name component."""
    return f"{len(symbol_prefix)}{symbol_prefix}"


def _needs_rename(demangled: str) -> bool:
    return _NEEDS_RENAME_RE.search(demangled) is not None


def _split_itanium_name(name: str) -> tuple[str, str] | None:
    for prefix in ITANIUM_MANGLED_PREFIXES:
        if name.startswith(prefix):
            return prefix, name[len(prefix) :]
    return None


def _renamed(name: str, demangled: str, component: str) -> str | None:
    """Returns the renamed symbol, or None when the symbol is out of scope."""
    split_name = _split_itanium_name(name)
    if split_name is None:
        return None
    if not _needs_rename(demangled):
        return None

    prefix, body = split_name
    marked_nested = _MARKED_NESTED_RE.match(body)
    if marked_nested:
        head = marked_nested.group(0)
        return prefix + head + component + body[len(head) :]
    return prefix + component + body


def _demangle(cxxfilt: str, names: list[str]) -> list[str]:
    """Demangles names 1:1 via llvm-cxxfilt; failures echo the input name."""
    if not names:
        return []
    result = subprocess.run(
        [cxxfilt],
        input="\n".join(names) + "\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    demangled = result.stdout.splitlines()
    if len(demangled) != len(names):
        raise RuntimeError(
            f"{cxxfilt} returned {len(demangled)} lines for {len(names)} symbols"
        )
    return demangled


def _write_rename_map(renames: set[tuple[str, str]], out_path: str) -> None:
    # Bazel actions write to unique output paths, so a plain write suffices.
    with open(out_path, "w", encoding="utf-8") as f:
        for old, new in sorted(renames):
            f.write(f"{old} {new}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an llvm-objcopy --redefine-syms file for IREE ABI renaming.",
    )
    parser.add_argument("--nm", required=True, help="Path to llvm-nm.")
    parser.add_argument("--cxxfilt", required=True, help="Path to llvm-cxxfilt.")
    parser.add_argument(
        "--input", required=True, help="Object file or static archive to inspect."
    )
    parser.add_argument("--out", required=True, help="Output rename-map path.")
    parser.add_argument(
        "--symbol-prefix",
        required=True,
        help="Logical ABI prefix (e.g. IREE18); inserted as a length-prefixed "
        "Itanium component.",
    )
    parser.add_argument(
        "--defined-only",
        action="store_true",
        help="Pass --defined-only to llvm-nm before generating the map.",
    )
    args = parser.parse_args()

    # The map keys must be the raw assembler-level names, so nm must not
    # demangle; demangling for the rename decision happens separately.
    nm_args = [args.nm, "--no-demangle"]
    if args.defined_only:
        nm_args.append("--defined-only")
    nm_args.append(args.input)

    result = subprocess.run(
        nm_args,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        # Unsupported inputs, corrupt archives, or toolchain mismatches should
        # fail the Bazel action with llvm-nm's own diagnostic.
        sys.stderr.write(result.stderr)
        return result.returncode

    mangled_names = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[-1]
        if _split_itanium_name(name) is not None:
            mangled_names.add(name)

    ordered_names = sorted(mangled_names)
    demangled_names = _demangle(args.cxxfilt, ordered_names)

    component = _itanium_component(args.symbol_prefix)
    renames: set[tuple[str, str]] = set()
    for name, demangled in zip(ordered_names, demangled_names):
        new_name = _renamed(name, demangled, component)
        if new_name:
            renames.add((name, new_name))

    _write_rename_map(renames, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

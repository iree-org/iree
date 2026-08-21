#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Export invariants of libIREECompiler.

The single renamed compiler library relies on naming conventions to keep three
symbol families apart:

  1. snake_case `iree_*`  -- the bundled runtime C-API; must be hidden so it
     can never shadow a co-resident iree.runtime.
  2. camelCase `ireeCompiler*` -- the compiler embedding C-API; must stay
     exported.
  3. `_ZN6IREE18*` / `_Z6IREE18*` -- the renamed llvm/mlir C++ internals used
     by dynamic plugins; must stay exported.

A convention violation (a snake_case export sneaking into the compiler, or the
hide-script over-matching) silently binds the wrong symbol at load time, so
this test turns it into a CI failure instead.

Extension point: a co-installed runtime library can be added to `data` and
checked for an empty exported-name intersection with the compiler library.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unittest

NM: str = ""
LIBRARY: str = ""

# Mach-O adds a leading underscore to C-level names.
_RUNTIME_CAPI_RE = re.compile(r"^_?iree_[a-z0-9_]+$")
_COMPILER_CAPI_RE = re.compile(r"^_?ireeCompiler[A-Za-z0-9]+$")
_RENAMED_CXX_RE = re.compile(r"^_?_Z(N|TVN|TTN|TIN|TSN|GVN)?6IREE18")


def _exported_symbols() -> list[str]:
    result = subprocess.run(
        [NM, "--extern-only", "--defined-only", "--no-demangle", LIBRARY],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    symbols = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[-2] not in ("U",):
            symbols.append(fields[-1])
    return symbols


class ExportedSymbolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.symbols = _exported_symbols()

    def test_runtime_capi_is_hidden(self):
        leaked = [s for s in self.symbols if _RUNTIME_CAPI_RE.match(s)]
        self.assertEqual(
            leaked[:20],
            [],
            msg="snake_case iree_* runtime C-API symbols leaked into the "
            "compiler export table; hide_iree_runtime_capi.* must cover them",
        )

    def test_compiler_capi_is_exported(self):
        self.assertTrue(
            any(_COMPILER_CAPI_RE.match(s) for s in self.symbols),
            msg="no ireeCompiler* embedding C-API symbol exported",
        )

    def test_renamed_internals_are_exported(self):
        self.assertTrue(
            any(_RENAMED_CXX_RE.match(s) for s in self.symbols),
            msg="no renamed (6IREE18) llvm/mlir C++ symbol exported; the "
            "rename pipeline or export scripts are broken",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nm", required=True)
    parser.add_argument("--library", required=True)
    args, remaining = parser.parse_known_args()
    NM = args.nm
    LIBRARY = args.library
    unittest.main(argv=[sys.argv[0]] + remaining)

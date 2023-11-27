// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// See llvm-project/lld/tools/lld/lld.cpp. Much of that is scaffolding for
// supporting symlink based lld which auto-detects the flavor. Instead, we
// duplicate the flavor parsing and invoke the backend directly similar to
// what the lldMain() does.

#if defined(IREE_COMPILER_LLD_DISABLED) ||                                     \
    (defined(IREE_COMPILER_LLD_ELF_DISABLED) &&                                \
     defined(IREE_COMPILER_LLD_COFF_DISABLED) &&                               \
     defined(IREE_COMPILER_LLD_MACHO_DISABLED) &&                              \
     defined(IREE_COMPILER_LLD_WASM_DISABLED))
#include <stdio.h>

#include "iree/compiler/tool_entry_points_api.h"

int ireeCompilerRunLldMain(int argc, char **argv) {
  fprintf(stderr, "IREE was not built with support for LLD\n");
  return 1;
}
#else
#include <cstdlib>
#include <vector>

#include "iree/compiler/tool_entry_points_api.h"
#include "lld/Common/Driver.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CrashRecoveryContext.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

using namespace lld;
using namespace llvm;
using namespace llvm::sys;

LLD_HAS_DRIVER(coff)
LLD_HAS_DRIVER(elf)
LLD_HAS_DRIVER(macho)
LLD_HAS_DRIVER(wasm)

[[noreturn]] static void die(const Twine &s) {
  llvm::errs() << s << "\n";
  exit(1);
}

static Flavor getFlavor(StringRef s) {
  return StringSwitch<Flavor>(s)
      .CasesLower("ld", "ld.lld", "gnu", Gnu)
      .CasesLower("wasm", "ld-wasm", Wasm)
      .CaseLower("link", WinLink)
      .CasesLower("ld64", "ld64.lld", "darwin", "darwinnew",
                  "ld64.lld.darwinnew", Darwin)
      .Default(Invalid);
}

static Flavor parseFlavor(std::vector<const char *> &v) {
  // Parse -flavor option.
  if (v.size() > 1 && v[1] == StringRef("-flavor")) {
    if (v.size() <= 2)
      die("missing arg value for '-flavor'");
    Flavor f = getFlavor(v[2]);
    if (f == Invalid)
      die("Unknown flavor: " + StringRef(v[2]));
    v.erase(v.begin() + 1, v.begin() + 3);
    return f;
  }
  die("Expected -flavor <gnu|link|darwin|wasm>");
}

int ireeCompilerRunLldMain(int argc, char **argv) {
  llvm::setBugReportMsg(
      "Please report issues to https://github.com/openxla/iree/issues and "
      "include the crash backtrace.\n");
  InitLLVM x(argc, argv);
  sys::Process::UseANSIEscapeCodes(true);
  bool exitEarly = true;
  bool disableOutput = false;
  llvm::raw_ostream &stdoutOS = llvm::outs();
  llvm::raw_ostream &stderrOS = llvm::errs();

  std::vector<const char *> args(argv, argv + argc);
  switch (parseFlavor(args)) {
  case Gnu:
#ifndef IREE_COMPILER_LLD_ELF_DISABLED
    return !elf::link(args, stdoutOS, stderrOS, exitEarly, disableOutput);
#else
    die("lld is not compiled with ELF support");
#endif
  case WinLink:
#ifndef IREE_COMPILER_LLD_COFF_DISABLED
    return !coff::link(args, stdoutOS, stderrOS, exitEarly, disableOutput);
#else
    die("lld is not compiled with COFF support");
#endif
  case Darwin:
#ifndef IREE_COMPILER_LLD_MACHO_DISABLED
    return !macho::link(args, stdoutOS, stderrOS, exitEarly, disableOutput);
#else
    die("lld is not compiled with MachO support");
#endif
  case Wasm:
#ifndef IREE_COMPILER_LLD_WASM_DISABLED
    return !lld::wasm::link(args, stdoutOS, stderrOS, exitEarly, disableOutput);
#else
    die("lld is not compiled with WASM support");
#endif
  default:
    die("lld is a generic driver.\n"
        "Invoke ld.lld (Unix), ld64.lld (macOS), lld-link (Windows), wasm-ld"
        " (WebAssembly) instead");
  }
}
#endif // IREE_COMPILER_LLD_DISABLED

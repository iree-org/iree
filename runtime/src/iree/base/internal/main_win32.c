// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdlib.h>

#include "iree/base/internal/main.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <combaseapi.h>

// Entry point when using /SUBSYSTEM:CONSOLE is the standard main().
int main(int argc, char** argv) { return iree_main(argc, argv); }

// Entry point when using /SUBSYSTEM:WINDOWS.
// https://docs.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-winmain
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance,
                   LPSTR lpCmdLine, int nShowCmd) {
  // Setup COM on the main thread.
  // NOTE: this may fail if COM has already been initialized - that's OK.
  CoInitializeEx(NULL, COINIT_MULTITHREADED);

  // Run standard main function.
  // We use the MSVCRT __argc/__argv to get access to the standard argc/argv
  // vs. using the flattened string passed to WinMain (that would require
  // complex unicode splitting/etc).
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/argc-argv-wargv
  return iree_main(__argc, __argv);
}

#endif  // IREE_PLATFORM_WINDOWS

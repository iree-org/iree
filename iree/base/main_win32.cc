// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>
#include <vector>

#include "iree/base/logging.h"
#include "iree/base/main.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

#include <combaseapi.h>
#include <shellapi.h>

namespace iree {
namespace {

// Entry point when using /SUBSYSTEM:CONSOLE is the standard main().
extern "C" int main(int argc, char** argv) { return IreeMain(argc, argv); }

// Entry point when using /SUBSYSTEM:WINDOWS.
extern "C" int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
  // Convert command line to an argv-like format.
  // NOTE: the command line that comes in with the WinMain arg is garbage.
  int argc = 0;
  wchar_t** argv_w = ::CommandLineToArgvW(::GetCommandLineW(), &argc);
  if (!argc || !argv_w) {
    LOG(FATAL) << "Unable to parse command line";
    return 1;
  }

  // Convert all args to narrow char strings.
  std::vector<std::string> allocated_strings(argc);
  std::vector<char*> argv_a(argc);
  for (int i = 0; i < argc; ++i) {
    size_t char_length = wcslen(argv_w[i]);
    allocated_strings[i].resize(char_length);
    argv_a[i] = const_cast<char*>(allocated_strings[i].data());
    std::wcstombs(argv_a[i], argv_w[i], char_length + 1);
  }
  ::LocalFree(argv_w);

  // Setup COM on the main thread.
  // NOTE: this may fail if COM has already been initialized - that's OK.
  ::CoInitializeEx(nullptr, COINIT_MULTITHREADED);

  // Run standard main function.
  int exit_code = IreeMain(argc, argv_a.data());

  // Release arg memory.
  argv_a.clear();
  allocated_strings.clear();

  return exit_code;
}

}  // namespace
}  // namespace iree

#endif  // IREE_PLATFORM_WINDOWS

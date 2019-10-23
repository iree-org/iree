// Copyright 2019 Google LLC
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

#include "bindings/python/pyiree/initialize.h"

#include <string.h>

#include <mutex>  // NOLINT

#include "base/init.h"

namespace iree {
namespace python {

namespace {

void InternalInitialize(const std::vector<std::string>& arguments) {
  int argc = arguments.size() + 1;  // plus one for program name.
  char** argv = static_cast<char**>(
      malloc(sizeof(char*) * (argc + 1)));  // plus one for null terminator.
  char** orig_argv = argv;
  argv[0] = strdup("<python_extension>");
  for (int i = 1; i < argc; ++i) {
    argv[i] = strdup(arguments[i - 1].c_str());
  }
  argv[argc] = nullptr;
  InitializeEnvironment(&argc, &argv);
  for (int i = 0; i < argc; ++i) {
    free(argv[i]);
  }
  free(orig_argv);
}

}  // namespace

void InitializeExtension(const std::vector<std::string>& arguments) {
  static std::once_flag init_once;
  std::call_once(init_once, InternalInitialize, arguments);
}

}  // namespace python
}  // namespace iree

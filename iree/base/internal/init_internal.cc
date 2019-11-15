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

#include "iree/base/internal/init_internal.h"

#include <string.h>

#include <set>

#include "absl/flags/parse.h"
#include "iree/base/initializer.h"

namespace iree {

void InitializeEnvironment(int* argc, char*** argv) {
  if (argc != nullptr && argv != nullptr && *argc != 0) {
    auto positional_args = absl::ParseCommandLine(*argc, *argv);
    if (positional_args.size() < *argc) {
      // Edit the passed argument refs to only include positional args.
      *argc = positional_args.size();
      for (int i = 0; i < *argc; ++i) {
        (*argv)[i] = positional_args[i];
      }
      (*argv)[*argc + 1] = nullptr;
    }
  }

  IREE_RUN_MODULE_INITIALIZERS();
}

}  // namespace iree

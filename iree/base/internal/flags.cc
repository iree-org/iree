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

#include "iree/base/internal/flags.h"

#include <stdlib.h>
#include <string.h>

// TODO(#3814): replace abseil with pretty much anything else.
#include "absl/flags/parse.h"

iree_status_t iree_flags_parse(int* argc, char*** argv) {
  if (argc == nullptr || argv == nullptr || *argc == 0) {
    // No flags; that's fine - in some environments flags aren't supported.
    return iree_ok_status();
  }

  auto positional_args = absl::ParseCommandLine(*argc, *argv);
  if (positional_args.size() < *argc) {
    // Edit the passed argument refs to only include positional args.
    *argc = static_cast<int>(positional_args.size());
    for (int i = 0; i < *argc; ++i) {
      (*argv)[i] = positional_args[i];
    }
    (*argv)[*argc + 1] = nullptr;
  }

  return iree_ok_status();
}

void iree_flags_parse_checked(int* argc, char*** argv) {
  iree_status_t status = iree_flags_parse(argc, argv);
  if (iree_status_is_cancelled(status)) {
    exit(EXIT_SUCCESS);
    return;
  }
  if (!iree_status_is_ok(status)) {
    // TODO(#2843): replace C++ logging.
    iree_status_ignore(status);
    exit(EXIT_FAILURE);
    return;
  }
}

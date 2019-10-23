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

#ifndef IREE_BASE_INIT_H_
#define IREE_BASE_INIT_H_

// Initializer macros are defined in separate files:
//   IREE_DECLARE_MODULE_INITIALIZER(name)
//   IREE_REGISTER_MODULE_INITIALIZER(name, body)
//   IREE_REGISTER_MODULE_INITIALIZER_SEQUENCE(name1, name2)
//   IREE_REQUIRE_MODULE_INITIALIZED(name)
//   IREE_RUN_MODULE_INITIALIZERS()
//   IREE_REQUIRE_MODULE_LINKED(name)
//
// These macros allow for arranging pieces of initialization code to be
// executed at a well-defined time and in a well-defined order.
//
// Initialization happens automatically during InitializeEnvironment(), which
// should be called early in main(), before other code runs.

#ifdef IREE_CONFIG_GOOGLE_INTERNAL
#include "iree/base/google/init_google.h"
#else
#include "iree/base/internal/init_internal.h"
#endif  // IREE_CONFIG_GOOGLE_INTERNAL

namespace iree {

// Initializes the system environment in a binary.
//
// This first parses command line flags, then resolves module initializers
// by calling IREE_RUN_MODULE_INITIALIZERS().
//
// 'argc' and 'argv' are the command line flags to parse.
//
// This should typically be called early in main(), before other code runs.
void InitializeEnvironment(int* argc, char*** argv);

}  // namespace iree

#endif  // IREE_BASE_INIT_H_

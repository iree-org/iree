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

#ifndef IREE_BASE_INTERNAL_FLAGS_H_
#define IREE_BASE_INTERNAL_FLAGS_H_

#include "iree/base/api.h"

//===----------------------------------------------------------------------===//
// Flag parsing
//===----------------------------------------------------------------------===//

// Parses flags from the given command line arguments.
// All flag-style arguments ('--foo', '-f', etc) will be consumed and argc/argv
// will be updated to contain only the program name (index 0) and any remaining
// positional arguments.
//
// Returns success if all flags were parsed and execution should continue.
// May return IREE_STATUS_CANCELLED if execution should be cancelled gracefully
// such as when --help is used.
//
// Usage:
//   extern "C" int main(int argc, char** argv) {
//     iree_status_t status = iree_flags_parse(&argc, &argv);
//     if (iree_status_is_cancelled(status)) return 0;
//     if (!iree_status_is_ok(status)) {
//       // TODO(#2843): replace C++ logging.
//       LOG(ERROR) << status;
//       iree_status_ignore(status);
//       return 1;
//     }
//     consume_positional_args(argc, argv);
//     return 0;
//   }
//
// Example:
//   argc = 4, argv = ['program', 'abc', '--flag=2', '-p']
// Results:
//   argc = 2, argv = ['program', 'abc']
iree_status_t iree_flags_parse(int* argc, char*** argv);

// Parses flags as with iree_flags_parse but will use exit() or abort().
// WARNING: this almost always what you want in a command line tool and *never*
// what you want when embedded in a host process. You don't want to have a flag
// typo and shut down your entire server/sandbox/Android app/etc.
void iree_flags_parse_checked(int* argc, char*** argv);

#endif  // IREE_BASE_INTERNAL_FLAGS_H_

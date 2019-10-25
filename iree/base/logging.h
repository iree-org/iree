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

#ifndef IREE_BASE_LOGGING_H_
#define IREE_BASE_LOGGING_H_

// Logging macros live in their own file so that we can use external versions
// as required.
//
// LOG(severity) << ...;
//   Logs a message at the given severity.
//   Severity:
//     INFO    Logs information text.
//     WARNING Logs a warning.
//     ERROR   Logs an error.
//     FATAL   Logs an error and exit(1).
//
// DLOG(severity) << ...;
//   Behaves like `LOG` in debug mode (i.e. `#ifndef NDEBUG`).
//   Otherwise, it compiles away and does nothing.
//
// VLOG(level) << ...;
//   Logs a verbose message at the given verbosity level.
//
// DVLOG(level) << ...;
//   Behaves like `VLOG` in debug mode (i.e. `#ifndef NDEBUG`).
//   Otherwise, it compiles away and does nothing.
//
// CHECK(condition) << ...;
//   Runtime asserts that the given condition is true even in release builds.
//   It's recommended that DCHECK is used instead as too many CHECKs
//   can impact performance.
//
// CHECK_EQ|NE|LT|GT|LE|GE(val1, val2) << ...;
//   Runtime assert the specified operation with the given values.
//
// DCHECK(condition) << ...;
//   Runtime asserts that the given condition is true only in non-opt builds.
//
// DCHECK_EQ|NE|LT|GT|LE|GE(val1, val2) << ...;
//   Runtime assert the specified operation with the given values in non-opt
//   builds.
//
// QCHECK(condition) << ...;
// QCHECK_EQ|NE|LT|GT|LE|GE(val1, val2) << ...;
//   These behave like `CHECK` but do not print a full stack trace.
//   They are useful when problems are definitely unrelated to program flow,
//   e.g. when validating user input.

#ifdef IREE_CONFIG_GOOGLE_INTERNAL
#include "iree/base/google/logging_google.h"
#else
#include "iree/base/internal/logging.h"
#endif  // IREE_CONFIG_GOOGLE_INTERNAL

#endif  // IREE_BASE_LOGGING_H_

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

#ifndef IREE_RT_DEBUG_DEBUG_SERVER_FLAGS_H_
#define IREE_RT_DEBUG_DEBUG_SERVER_FLAGS_H_

#include "iree/base/status.h"
#include "iree/rt/debug/debug_server.h"

namespace iree {
namespace rt {
namespace debug {

// Creates a debug server based on the current --iree_* debug flags.
// Returns nullptr if no server is compiled in or the flags are not set.
StatusOr<std::unique_ptr<DebugServer>> CreateDebugServerFromFlags();

}  // namespace debug
}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_DEBUG_DEBUG_SERVER_FLAGS_H_

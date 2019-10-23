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

#ifndef IREE_HAL_EXECUTABLE_H_
#define IREE_HAL_EXECUTABLE_H_

#include "iree/hal/resource.h"

namespace iree {
namespace hal {

class Executable : public Resource {
 public:
  ~Executable() override = default;

  // True if the executable was prepared with debugging enabled and the device
  // and input data support debugging (symbols present, etc).
  virtual bool supports_debugging() const = 0;

  // TODO(benvanik): disassembly methods.

  // TODO(benvanik): relative offset calculation:
  //   - step once
  //   - step over
  //   - step out

  // TODO(benvanik): create executable split on breakpoint.
  // Executable should return when the breakpoint is hit without any future
  // modifications to output buffers. If the breakpoint is not hit the
  // executable should run to completion as normal.

  // TODO(benvanik): retrieve coverage info.
  // Returns a buffer containing offset -> coverage metrics. Note that depending
  // on the device this may only contain a single coverage metric for the entire
  // executable or some subset of the available offsets.

  // TODO(benvanik): retrieve profiling info.

 protected:
  Executable() = default;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_EXECUTABLE_H_

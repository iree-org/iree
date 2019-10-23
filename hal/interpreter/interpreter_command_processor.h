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

#ifndef IREE_HAL_INTERPRETER_INTERPRETER_COMMAND_PROCESSOR_H_
#define IREE_HAL_INTERPRETER_INTERPRETER_COMMAND_PROCESSOR_H_

#include "hal/host/host_local_command_processor.h"

namespace iree {
namespace hal {

class InterpreterCommandProcessor final : public HostLocalCommandProcessor {
 public:
  InterpreterCommandProcessor(Allocator* allocator,
                              CommandBufferModeBitfield mode,
                              CommandCategoryBitfield command_categories);
  ~InterpreterCommandProcessor() override;

  Status Dispatch(const DispatchRequest& dispatch_request) override;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_INTERPRETER_INTERPRETER_COMMAND_PROCESSOR_H_

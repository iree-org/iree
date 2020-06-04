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
//
#ifndef IREE_HAL_DYLIB_DYLIB_COMMAND_PROCESSOR_H_
#define IREE_HAL_DYLIB_DYLIB_COMMAND_PROCESSOR_H_

#include "iree/hal/host/host_local_command_processor.h"

namespace iree {
namespace hal {
namespace dylib {

class DyLibCommandProcessor final : public HostLocalCommandProcessor {
 public:
  explicit DyLibCommandProcessor(CommandCategoryBitfield command_categories);
  ~DyLibCommandProcessor() override;

  Status DispatchInline(
      Executable* executable, int32_t entry_point,
      std::array<uint32_t, 3> workgroups,
      const PushConstantBlock& push_constants,
      absl::Span<const absl::Span<const DescriptorSet::Binding>> set_bindings)
      override;
};

}  // namespace dylib
}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_DYLIB_DYLIB_COMMAND_PROCESSOR_H_

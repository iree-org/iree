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

#ifndef IREE_HAL_HOST_HOST_EXECUTABLE_H_
#define IREE_HAL_HOST_HOST_EXECUTABLE_H_

#include "iree/base/status.h"
#include "iree/hal/descriptor_set.h"
#include "iree/hal/executable.h"

namespace iree {
namespace hal {

// Computed push constant values available to all tiles in the grid.
struct PushConstantBlock {
  // We limit ourselves to 32 constants (32*sizeof(uint32) = 128b).
  // This is the lower bound for Vulkan implementations and ensures that we
  // have consistent support everywhere.
  std::array<uint32_t, 32> values;
};

// Abstract host-local executable that can dispatch grid-based tiles.
// Implementations provide the logic to process individual tiles within the
// workgroup-defined XYZ grid.
//
// Thread-safe; the processor may be called to process the grid by any thread in
// any order.
class HostExecutable : public Executable {
 public:
  // Grid parameters shared for all tiles within a dispatch.
  struct DispatchParams {
    // Entry point within the executable.
    int32_t entry_point = 0;

    // Total workgroup XYZ count for the grid.
    std::array<uint32_t, 3> workgroup_count;

    // Push constants populated by the command buffer.
    const PushConstantBlock* push_constants = nullptr;

    // Descriptor set bindings organized by set and binding ordinal.
    absl::Span<const absl::Span<const DescriptorSet::Binding>> set_bindings;
  };

  struct DispatchState : public RefObject<DispatchState> {
    virtual ~DispatchState() = default;
  };

  // Begins processing a grid dispatch with the given parameters.
  // May be called from any thread. Returns dispatch state that will be passed
  // to all DispatchTile calls from the same dispatch operation.
  virtual StatusOr<ref_ptr<DispatchState>> PrepareDispatch(
      const DispatchParams& params) = 0;

  // Processes a single tile within the grid.
  // |workgroup_xyz| is the tile coordinates in the grid as defined during
  // preparation. May be called from any thread.
  virtual Status DispatchTile(DispatchState* state,
                              std::array<uint32_t, 3> workgroup_xyz) = 0;

 protected:
  HostExecutable() = default;
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_HOST_HOST_EXECUTABLE_H_

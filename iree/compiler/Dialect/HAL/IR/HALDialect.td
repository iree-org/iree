// Copyright 2021 Google LLC
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

#ifndef IREE_DIALECT_HAL_DIALECT
#define IREE_DIALECT_HAL_DIALECT

include "iree/compiler/Dialect/IREE/IR/IREEBase.td"

//===----------------------------------------------------------------------===//
// IREE HAL (Hardware Abstraction Layer) dialect
//===----------------------------------------------------------------------===//

def HAL_Dialect : Dialect {
  let name = "hal";
  let cppNamespace = "::mlir::iree_compiler::IREE::HAL";

  let summary = [{
    A dialect representing operations against the IREE HAL.
  }];
  let description = [{
    This can be thought of as a Vulkan-like model with all of the graphics bits
    chopped out.

    The type set is limited to those that can be represented in the IREE HAL
    design: buffers and views, synchronization primitives like semaphores, and
    and command buffers. The intent is that if a device could implement the HAL
    interface the sequencer ops could run on that device, such as being able to
    run on a GPU via indirect command buffers.

    Though this is mostly a 1:1 mapping to the iree::hal API there are some
    methods omitted as they are not likely to be needed in IR. It's assumed that
    either sequencer interfaces will encapsulate the logic (such as device
    resolution) or that certain features are unsafe to expose to user-defined
    input.
  }];
}

#endif // IREE_DIALECT_HAL_DIALECT

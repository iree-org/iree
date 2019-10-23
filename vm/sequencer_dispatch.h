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

#ifndef IREE_VM_SEQUENCER_DISPATCH_H_
#define IREE_VM_SEQUENCER_DISPATCH_H_

#include "base/status.h"
#include "hal/buffer_view.h"
#include "hal/device_placement.h"
#include "rt/stack.h"
#include "rt/stack_frame.h"

namespace iree {
namespace vm {

// TODO(benvanik): API that supports yielding.
Status DispatchSequence(const hal::DevicePlacement& placement, rt::Stack* stack,
                        rt::StackFrame* entry_stack_frame,
                        absl::Span<hal::BufferView> entry_results);

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_SEQUENCER_DISPATCH_H_

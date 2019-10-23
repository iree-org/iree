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

#ifndef IREE_HAL_STACK_TRACE_H_
#define IREE_HAL_STACK_TRACE_H_

namespace iree {
namespace hal {

class StackTrace {
 public:
  // TODO(benvanik): define contents.
  //  frame:
  //    device type (cpu, etc)
  //    effective processor type (determines disasm/etc) <- r52, vliw, etc
  //    effective offset <- in disasm (abstract, could be op ordinal, byte
  //      offset)
  //    source offset <- used in source map lookup
  //    physical offset <- informative, void* (real memory address)
  // physical_context (x86 registers, etc)
  // effective_context (??)
  // source_context (buffer views/etc?)
};

}  // namespace hal
}  // namespace iree

#endif  // IREE_HAL_STACK_TRACE_H_

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

#ifndef THIRD_PARTY_MLIR_EDGE_IREE_VM_SEQUENCER_CONTEXT_H_
#define THIRD_PARTY_MLIR_EDGE_IREE_VM_SEQUENCER_CONTEXT_H_

#include <memory>
#include <vector>

#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/mlir_edge/iree/base/status.h"
#include "third_party/mlir_edge/iree/hal/buffer_view.h"
#include "third_party/mlir_edge/iree/vm/context.h"
#include "third_party/mlir_edge/iree/vm/function.h"
#include "third_party/mlir_edge/iree/vm/instance.h"
#include "third_party/mlir_edge/iree/vm/module.h"

namespace iree {
namespace vm {

class SequencerContext final : public Context {
 public:
  explicit SequencerContext(std::shared_ptr<Instance> instance);
  ~SequencerContext() override;

  Status RegisterNativeFunction(std::string name,
                                NativeFunction native_function) override;

  Status RegisterModule(std::unique_ptr<Module> module) override;

  // TODO(benvanik): helpers to make passing args easier
  Status Invoke(FiberState* fiber_state, vm::Function function,
                absl::Span<hal::BufferView> args,
                absl::Span<hal::BufferView> results) const;

 private:
  std::shared_ptr<Instance> instance_;
};

}  // namespace vm
}  // namespace iree

#endif  // THIRD_PARTY_MLIR_EDGE_IREE_VM_CONTEXT_H_

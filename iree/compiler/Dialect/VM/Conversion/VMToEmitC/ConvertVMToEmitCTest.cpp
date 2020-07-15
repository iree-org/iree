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

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/ConvertVMToEmitC.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

static PassRegistration<IREE::VM::ConvertVMToEmitCPass> pass(
    "test-iree-convert-vm-to-emitc", "Convert VM Ops to the EmitC VM dialect");

}  // namespace iree_compiler
}  // namespace mlir
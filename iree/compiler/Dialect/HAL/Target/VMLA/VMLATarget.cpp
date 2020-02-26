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

#include "iree/compiler/Dialect/HAL/Target/VMLA/VMLATarget.h"

#include "llvm/Support/CommandLine.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(benvanik): add flags.
// static llvm::cl::OptionCategory halVMLAOptionsCategory(
//     "IREE VMLA backend options");

VMLATargetOptions getVMLATargetOptionsFromFlags() {
  VMLATargetOptions targetOptions;
  // TODO(benvanik): flags.
  return targetOptions;
}

LogicalResult translateToVMLAExecutable(
    IREE::HAL::ExecutableOp executableOp,
    ExecutableTargetOptions executableOptions,
    VMLATargetOptions targetOptions) {
  return success();
}

static ExecutableTargetRegistration targetRegistration(
    "vmla", +[](IREE::HAL::ExecutableOp executableOp,
                ExecutableTargetOptions executableOptions) {
      return translateToVMLAExecutable(executableOp, executableOptions,
                                       getVMLATargetOptionsFromFlags());
    });

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

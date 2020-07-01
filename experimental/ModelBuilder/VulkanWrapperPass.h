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

#ifndef IREE_EXPERIMENTAL_MODELBUILDER_VULKANWRAPPERPASS_H_
#define IREE_EXPERIMENTAL_MODELBUILDER_VULKANWRAPPERPASS_H_

#include <memory>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace spirv {
class FuncOp;
}

class ModuleOp;
template <typename T>
class OperationPass;
/// Create a c interface function wrapping a vulkan dispatch for the existing
/// GPU module.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAddVulkanLaunchWrapperPass(
    llvm::ArrayRef<int64_t> workloadSize, llvm::ArrayRef<Type> args);

/// Set SPIRV ABI for kernel arguments. This hardcode the binding information
/// to be able to wok with vulkan runner.
std::unique_ptr<OperationPass<spirv::FuncOp>> createSetSpirvABIPass();
}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_MODELBUILDER_VULKANWRAPPERPASS_H_

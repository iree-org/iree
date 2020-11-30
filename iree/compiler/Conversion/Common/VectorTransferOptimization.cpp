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

#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct VectorTransferOptimizationPass
    : public PassWrapper<VectorTransferOptimizationPass, FunctionPass> {
  void runOnFunction() override { vector::transferOpflowOpt(getFunction()); }
};

}  // namespace

std::unique_ptr<FunctionPass> createVectorTransferOptimizationPass() {
  return std::make_unique<VectorTransferOptimizationPass>();
}

static PassRegistration<VectorTransferOptimizationPass> pass(
    "iree-codegen-optimize-vector-transfer",
    "Run optimization transformations on vector transfer operations",
    [] { return std::make_unique<VectorTransferOptimizationPass>(); });

}  // namespace iree_compiler
}  // namespace mlir

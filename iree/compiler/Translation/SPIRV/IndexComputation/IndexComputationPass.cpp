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

//===- IndexComputationPass.cpp --------------------------------*- C++//-*-===//
//
// Pass to perform index propagation in iree dispatch functions
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/IndexComputation/IREEIndexComputation.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

class IndexComputationPass : public FunctionPass<IndexComputationPass> {
  void runOnFunction() override;
};

}  // namespace

void IndexComputationPass::runOnFunction() {
  // Initialize the index computation.
  IndexPropagationList<
      IndexPropagationOp<ConstantOp>, ExtractElementOpIndexPropagation,
      IndexPropagationOp<ReturnOp>,
      // IREE-specific ops:
      IREELoadIndexPropagation, IREEStoreIndexPropagation,
      IREEStoreReduceIndexPropagation,
      // Standard dialect unary elementwise ops:
      NoBroadcastPwOpIndexPropagation<SIToFPOp>,
      NoBroadcastPwOpIndexPropagation<SignExtendIOp>,
      // Standard dialect binary elementwise ops:
      NoBroadcastPwOpIndexPropagation<AddFOp>,
      NoBroadcastPwOpIndexPropagation<AddIOp>,
      NoBroadcastPwOpIndexPropagation<AndOp>,
      NoBroadcastPwOpIndexPropagation<CmpFOp>,
      NoBroadcastPwOpIndexPropagation<CmpIOp>,
      NoBroadcastPwOpIndexPropagation<DivFOp>,
      NoBroadcastPwOpIndexPropagation<SignedDivIOp>,
      NoBroadcastPwOpIndexPropagation<UnsignedDivIOp>,
      NoBroadcastPwOpIndexPropagation<MulFOp>,
      NoBroadcastPwOpIndexPropagation<MulIOp>,
      NoBroadcastPwOpIndexPropagation<OrOp>,
      NoBroadcastPwOpIndexPropagation<RemFOp>,
      NoBroadcastPwOpIndexPropagation<SignedRemIOp>,
      NoBroadcastPwOpIndexPropagation<UnsignedRemIOp>,
      NoBroadcastPwOpIndexPropagation<SubFOp>,
      NoBroadcastPwOpIndexPropagation<SubFOp>,
      NoBroadcastPwOpIndexPropagation<SubIOp>,
      NoBroadcastPwOpIndexPropagation<TruncateIOp>,
      NoBroadcastPwOpIndexPropagation<XOrOp>,
      NoBroadcastPwOpIndexPropagation<ZeroExtendIOp>,
      // XLA unary elementwise ops:
      NoBroadcastPwOpIndexPropagation<xla_hlo::AbsOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::CeilOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::ConvertOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::CosOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::SinOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::ExpOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::FloorOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::LogOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::NegOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::RsqrtOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::SignOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::SqrtOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::TanhOp>,
      // XLA binary elementwise ops:
      NoBroadcastPwOpIndexPropagation<xla_hlo::AddOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::AndOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::DivOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::MaxOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::MinOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::MulOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::SubOp>,
      // XLA other ops:
      // TODO(ravishankarm): conv, dot.
      // TODO(ravishankarm): gather.
      // TODO(ravishankarm): pad.
      // TODO(hanchung): dynamic_slice.
      NoBroadcastPwOpIndexPropagation<xla_hlo::CopyOp>,
      ReshapeOpIndexPropagation<xla_hlo::ReshapeOp>,
      NoBroadcastPwOpIndexPropagation<xla_hlo::SelectOp>,
      XLABroadcastOpIndexPropagation, XLABroadcastInDimOpIndexPropagation,
      XLAConcatenateOpIndexPropagation, XLAGatherOpIndexPropagation,
      XLAPadOpIndexPropagation, XLAReverseOpIndexPropagation,
      XLASliceOpIndexPropagation, XLATransposeOpIndexPropagation>
      indexPropagation;

  auto funcOp = getFunction();
  if (!funcOp.getAttr("iree.executable.export")) return;
  if (failed(indexPropagation.propagate(funcOp.getBody()))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<FuncOp>> createIndexComputationPass() {
  return std::make_unique<IndexComputationPass>();
}
static PassRegistration<IndexComputationPass> pass(
    "iree-index-computation",
    "Index propagation within IREE dispatch functions");

}  // namespace iree_compiler
}  // namespace mlir

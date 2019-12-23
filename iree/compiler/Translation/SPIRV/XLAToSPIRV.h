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

//===- XLAToSPIRV.h --------------------------------------------*- C++//-*-===//
//
// SPIR-V Code-generation for xla_hlo operations within IREE Dispatch functions
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_H
#define IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_H

#include "iree/compiler/Translation/SPIRV/SPIRVLowering.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

/// Lowers a xla_hlo.concatenate_op to SPIR-V. All the operands are assumed to
/// be shifted to right indices by index propagation.
class XLAConcatenateOpSPIRVLowering final
    : public SPIRVOpLowering<xla_hlo::ConcatenateOp> {
 public:
  using SPIRVOpLowering<xla_hlo::ConcatenateOp>::SPIRVOpLowering;
  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

/// Lowers a xla_hlo.convert op to SPIR-V. If the source type is as same as the
/// destination type, it will be a NOP.
class XLAConvertOpSPIRVLowering final
    : public SPIRVOpLowering<xla_hlo::ConvertOp> {
 public:
  using SPIRVOpLowering<xla_hlo::ConvertOp>::SPIRVOpLowering;
  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

/// Lowers a xla_hlo.gather_op to SPIR-V.
class XLAGatherOpSPIRVLowering final
    : public SPIRVOpLowering<xla_hlo::GatherOp> {
 public:
  using SPIRVOpLowering<xla_hlo::GatherOp>::SPIRVOpLowering;
  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

/// Lowers a xla_hlo.pad_op to SPIR-V. Based on the value of the result index
/// computed at a thread, if the index falls in a padding location, the padding
/// value is to be used. If not use the value obtained from the tensor operand.
class XLAPadOpSPIRVLowering final : public SPIRVOpLowering<xla_hlo::PadOp> {
 public:
  using SPIRVOpLowering<xla_hlo::PadOp>::SPIRVOpLowering;
  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_H

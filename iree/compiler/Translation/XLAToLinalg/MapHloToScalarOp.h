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

#ifndef IREE_COMPILER_TRANSLATION_XLATOLINALG_MAPHLOTOSCALAROP_H_
#define IREE_COMPILER_TRANSLATION_XLATOLINALG_MAPHLOTOSCALAROP_H_

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

// ScalarOp::FOp maps an HLO op to the corresponding std op for floating point,
// and ScalarOp::IOp maps an HLO op to the corresponding std op for integer.
template <typename HLO_BinaryOp>
struct ScalarOp;

template <typename HLO_BinaryOp>
using ScalarFOp = typename ScalarOp<HLO_BinaryOp>::FOp;
template <typename HLO_BinaryOp>
using ScalarIOp = typename ScalarOp<HLO_BinaryOp>::IOp;

template <>
struct ScalarOp<xla_hlo::AddOp> {
  using FOp = ::mlir::AddFOp;
  using IOp = ::mlir::AddIOp;
};
template <>
struct ScalarOp<xla_hlo::DivOp> {
  using FOp = ::mlir::DivFOp;
  using IOp = ::mlir::SignedDivIOp;
};
template <>
struct ScalarOp<xla_hlo::SubOp> {
  using FOp = ::mlir::SubFOp;
  using IOp = ::mlir::SubIOp;
};
template <>
struct ScalarOp<xla_hlo::MulOp> {
  using FOp = ::mlir::MulFOp;
  using IOp = ::mlir::MulIOp;
};

template <typename HloOp>
Operation* mapToStdScalarOp(HloOp hloOp, Type resultType,
                            ArrayRef<Value> blockArgs, OpBuilder builder) {
  if (resultType.isa<IntegerType>()) {
    return builder.template create<ScalarIOp<HloOp>>(hloOp.getLoc(), resultType,
                                                     blockArgs, mlir::None);
  }
  if (resultType.isa<FloatType>()) {
    return builder.template create<ScalarFOp<HloOp>>(hloOp.getLoc(), resultType,
                                                     blockArgs, mlir::None);
  }
  return nullptr;
}

template <>
inline Operation* mapToStdScalarOp<xla_hlo::ExpOp>(xla_hlo::ExpOp hloOp,
                                                   Type resultType,
                                                   ArrayRef<Value> blockArgs,
                                                   OpBuilder builder) {
  return resultType.isa<FloatType>()
             ? builder.create<::mlir::ExpOp>(hloOp.getLoc(), resultType,
                                             blockArgs, mlir::None)
             : nullptr;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_XLATOLINALG_MAPHLOTOSCALAROP_H_

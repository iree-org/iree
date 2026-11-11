// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_

#include <optional>

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "iree/compiler/GlobalOptimization/QuantizationUtils.h"
#include "llvm/ADT/APSInt.h"

namespace mlir {
class Type;
class Value;
class CastOpInterface;
class OpBuilder;
class Location;
class NamedAttribute;
class OpOperand;
class SymbolTableCollection;
class SymbolUserMap;
} // namespace mlir

namespace mlir::iree_compiler::IREE::Util {
class GlobalOpInterface;
} // namespace mlir::iree_compiler::IREE::Util

namespace mlir::iree_compiler::GlobalOptimization {

/// Returns a CastOpInterface op, if the producer is a CastOpInterface op, or a
/// linalg::GenericOp that performs only a CastOpInterface on its input.
/// The CastOpInterface op should extend the bitwidth of the source.
/// The bitwidth of the source element type should be greater than 1. If it is
/// casting from i1 types, a std::nullopt is returned. It is dangerous to mix
/// boalean concept and i1 subtypes concept at graph optimization level. We
/// ignore this type of casting ops intentionally.
/// TODO(hanchung): Remove the restriction about i1 after we can handle i1
/// sub-type emulation and deprecate TypePropagation pass.
///
/// If it is not from a casting op, it returns a std::nullopt.
///
/// **Note: If the CastOpInterface has been generalized, the return Operation
///         is the body CastOpInterface op, not the linalg::GenericOp.
std::optional<CastOpInterface> getDefiningNonI1ExtendingCastOp(Value input);

/// Returns the source element type of the defining CastOpInterface of `input`,
/// if there is one. Otherwise return std::nullopt.
std::optional<Type> getCastElemType(Value input);

/// Create an elementwise identity map linalg::GenericOp that casts the `input`
/// with the same cast operation as the passed CastOpInterface `castOp`.
Value createGenericElementwiseCastOp(
    OpBuilder &builder, Location loc, Value input, CastOpInterface castOp,
    ArrayRef<NamedAttribute> attrs,
    std::optional<IREE::Encoding::EncodingAttr> encoding = std::nullopt);

// Reduce the input value along the reduction dimensions.
Value sumReduceDimensionSubset(ImplicitLocOpBuilder &rewriter, Value val,
                               Type accETy, ArrayRef<bool> is_reduction);

/// Returns the global that `value` loads, or null if it is not a global load.
IREE::Util::GlobalOpInterface
getLoadedGlobal(Value value, SymbolTableCollection &symbolTables);

/// Returns the value that the `tensor.collapse_shape` and `tensor.expand_shape`
/// ops producing `value` read, or `value` itself if no such op produces it.
/// These reshapes keep the data unchanged.
Value getReshapeSource(Value value);

/// Returns every use that reads the data of `value`: reshapes are followed up
/// to their source and back down, across all loads when the source is a global
/// load. Returns nullopt if the global is public, so that code outside the
/// module can read it, or has users other than loads.
std::optional<SmallVector<OpOperand *>>
getUsesOfSameData(Value value, const SymbolUserMap &symbolUsers,
                  SymbolTableCollection &symbolTables);

/// Returns the integers `value` holds, read as unsigned if `isUnsigned`, when
/// it is a constant or a load of an immutable global that only loads use,
/// possibly through reshapes, or nullopt otherwise. A splat gives a single
/// value.
std::optional<SmallVector<APSInt>>
getConstantIntegerValues(Value value, bool isUnsigned,
                         const SymbolUserMap &symbolUsers,
                         SymbolTableCollection &symbolTables);

/// Returns the smallest range holding the integers `value` holds, read as
/// unsigned if `isUnsigned`, when it is a constant or a load of an immutable
/// global that only loads use, possibly through reshapes, or nullopt otherwise.
std::optional<QuantMinMax>
getConstantIntegerRange(Value value, bool isUnsigned,
                        const SymbolUserMap &symbolUsers,
                        SymbolTableCollection &symbolTables);

/// Returns true when `info` is a constant expression whose roots are inline
/// constants or loads of immutable globals that only loads use and that have
/// an inline initial value.
bool isComputedFromInlineData(
    const IREE::Util::ConstExprAnalysis::ConstValueInfo &info,
    const SymbolUserMap &symbolUsers, SymbolTableCollection &symbolTables);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_

#include <optional>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"

namespace mlir {
class Type;
class Value;
class CastOpInterface;
class OpBuilder;
class Location;
class NamedAttribute;
} // namespace mlir

namespace mlir::iree_compiler::GlobalOptimization {

/// Returns a CastOpInterface op, if the producer is a CastOpInterface op, or a
/// linalg::GenericOp that performs only a CastOpInterface on its input.
/// The CastOpInterface op should extend the bitwidth of the source.
/// The bitwidth of the source element type should be greater than 1. If it is
/// casting from i1 types, a std::nullopt is returned. It is dangerous to mix
/// boalean concept and i1 subtypes concept at graph optimizatoin level. We
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
    std::optional<IREE::LinalgExt::EncodingAttr> encoding = std::nullopt);

/// Creates a dispatch region out of a sequence of consecutive ops.
FailureOr<IREE::Flow::DispatchRegionOp>
wrapConsecutiveOpsInDispatchRegion(RewriterBase &rewriter,
                                   SmallVector<Operation *> ops);

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_UTILS_H_

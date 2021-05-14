// Copyright 2021 Google LLC
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

//===- LoweringConfig.h - Declares configuration for lowering Linalg ops --===//
//
// This file declares an attribute that drives how a dispatch region containing
// a set of operations are lowered. The attribute itself is attached to Linalg
// operations, and help converting a Linalg operation into "scalar code".
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CONVERSION_COMMON_LOWERINGCONFIG_H_
#define IREE_COMPILER_CONVERSION_COMMON_LOWERINGCONFIG_H_

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h.inc"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfigEnums.h.inc"
// clang-format on

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for getting/setting the `hal.lowering.*` attributes that drive the
// linalg-based lowering.
// ===----------------------------------------------------------------------===//

/// Returns the lowering configuration set for an operation.
IREE::HAL::LoweringConfig getLoweringConfig(Operation *op);

/// Returns true if an operation has a lowering configuration set.
bool hasLoweringConfig(Operation *op);

/// Sets the lowering configuration if one isnt already set and returns
/// true. Returns false if a configuration already exists.
bool setLoweringConfig(Operation *op, IREE::HAL::LoweringConfig config);

/// Removes the lowering configuration on the operation if it exists.
void eraseLoweringConfig(Operation *op);

//===----------------------------------------------------------------------===//
// Helpers for accessing values from the LoweringConfig attribute.
//===----------------------------------------------------------------------===//

// TODO(ravishankarm): Struct attributes dont have a way of defining extra class
// methods. When they do, these could all be moved into the attribute definition
// itself.

/// Stores the tile sizes to use at different levels of tiling as a vector of
/// vectors.
/// - First level tiling maps to workgroups.
/// - Second level tiling maps to subgroups.
/// - Third level tiling maps to invocations.
using TileSizesListType = SmallVector<SmallVector<int64_t, 4>, 1>;
using TileSizesListTypeRef = ArrayRef<SmallVector<int64_t, 4>>;

/// Construct a lowering configuration.
IREE::HAL::LoweringConfig getConfigAttr(TileSizesListTypeRef tileSizes,
                                        ArrayRef<int64_t> nativeVectorSize,
                                        MLIRContext *context);

/// Get the tile sizes for all levels.
TileSizesListType getTileSizes(IREE::HAL::LoweringConfig config);

/// Get the tile sizes for all levels for an operation if the lowering
/// configuration is set.
inline TileSizesListType getTileSizes(Operation *op) {
  auto configAttr = getLoweringConfig(op);
  if (!configAttr) return {};
  return getTileSizes(configAttr);
}

/// Get the tile sizes for level `level`, if it is defined. Returns {} if tile
/// sizes are not set for that level.
SmallVector<int64_t, 4> getTileSizes(IREE::HAL::LoweringConfig config,
                                     unsigned level);

/// Get the tile sizes for level `level` for an operation if the lowering
/// configuration for the operation is set, and tile sizes are defined for that
/// level.
inline SmallVector<int64_t, 4> getTileSizes(Operation *op, unsigned level) {
  auto configAttr = getLoweringConfig(op);
  if (!configAttr) return {};
  return getTileSizes(configAttr, level);
}
SmallVector<Value, 4> getTileSizes(OpBuilder &b, Operation *op, unsigned level);

/// Gets the native vector size defined in the lowering configuration.
SmallVector<int64_t, 4> getNativeVectorSize(IREE::HAL::LoweringConfig config);

/// Gets the native vector size defined for lowering an operation, if the
/// lowering configuration is defined. If not returns empty vector.
inline SmallVector<int64_t, 4> getNativeVectorSize(Operation *op) {
  auto configAttr = getLoweringConfig(op);
  if (!configAttr) return {};
  return getNativeVectorSize(configAttr);
}

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CONVERSION_COMMON_LOWERINGCONFIG_H_

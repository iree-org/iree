// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOTORCHATTENTION_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOTORCHATTENTION_H_

#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::stablehlo {
class TargetRegistry;

// Registers the version-1 torch.scaled_dot_product_attention composite, lowered
// to iree_linalg_ext.attention.
//
// Fully masked rows return zero. Excluded values are suppressed when all
// queries share the same excluded keys; query-dependent nonfinite values
// retain the attention contraction's floating-point behavior.
//
// Operands are query (B..., L, E), key (B..., S, E), value (B..., S, Ev) and an
// optional mask, boolean (true attends) or additive in the query type, shaped
// (L, S) or (B..., L, S). Attributes are is_causal and enable_gqa, both bool,
// and scale, a float that is absent for 1 / sqrt(E). Static head counts may
// use grouped-query attention with dynamic sequence lengths. A causal mask
// with an explicit mask and unsupported types or shapes keep the decomposition.
LogicalResult populateTorchAttentionTargetRegistry(TargetRegistry &registry);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOTORCHATTENTION_H_

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
// Known limitation: the lowering zeroes a row only when the mask hides it,
// whereas the producer zeroes a row whose scores are all -inf. The two differ
// for an unmasked -inf score row and for NaN scores under an all-false mask.
//
// Operands are query (B..., L, E), key (B..., S, E), value (B..., S, Ev) and an
// optional mask, boolean (true attends) or additive in the query type, shaped
// (L, S) or (B..., L, S). Attributes are is_causal and enable_gqa, both bool,
// and scale, a float that is absent for 1 / sqrt(E). Without explicit scale E
// must be static. Grouped-query attention requires statically provable head
// grouping. Causal attention supports dynamic sequence extents for f16, bf16
// and f32, with score exclusion after scaling, including zero/negative scales.
// Causal inputs of at most 8 bits and causal mode with an explicit mask keep
// their decomposition. Unsupported versions, types or profiles also decline
// without modifying IR.
LogicalResult populateTorchAttentionTargetRegistry(TargetRegistry &registry);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOTORCHATTENTION_H_

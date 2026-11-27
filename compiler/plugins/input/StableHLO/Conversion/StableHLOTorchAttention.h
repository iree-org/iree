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
// The lowering preserves the producer's additive masking and safe softmax:
// all -inf score rows have zero probabilities; NaN score rows propagate NaN.
//
// Operands are query (B..., L, E), key (B..., S, E), value (B..., S, Ev) and an
// optional mask, boolean (true attends) or additive in the query type, shaped
// (L, S) or (B..., L, S). Attributes are is_causal and enable_gqa, both bool,
// and scale, a float that is absent for 1 / sqrt(E). Without explicit scale E
// must be static. Grouped-query attention requires statically provable head
// grouping. Causal attention supports dynamic sequence extents for f16, bf16
// and f32, with mask bias after scaling, including zero/negative scales.
// Other element types and causal mode with an explicit mask keep their
// decomposition. Unsupported versions or profiles also decline
// without modifying IR.
LogicalResult populateTorchAttentionTargetRegistry(TargetRegistry &registry);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOTORCHATTENTION_H_

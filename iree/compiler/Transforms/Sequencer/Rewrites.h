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

#ifndef IREE_COMPILER_TRANSFORMS_SEQUENCER_REWRITES_H_
#define IREE_COMPILER_TRANSFORMS_SEQUENCER_REWRITES_H_

#include "iree/compiler/Utils/TypeConversionUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace iree_compiler {

// Adds rewrite patterns for lowering IREE Sequencer HL ops (iree_hl_seq.*)
// to LL ops (iree_ll_seq.*).
void populateSequencerLoweringPatterns(OwningRewritePatternList &patterns,
                                       MLIRContext *ctx);

// Adds rewrite patterns for lowering xla_hlo ops to Sequencer HL ops.
void populateLowerXlaToSequencerPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);

// Adds rewrite patterns for lowering standard ops to Sequencer HL ops.
void populateLowerStdToSequencerPatterns(OwningRewritePatternList &patterns,
                                         MLIRContext *ctx);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSFORMS_SEQUENCER_REWRITES_H_

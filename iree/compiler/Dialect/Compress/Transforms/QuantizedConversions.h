// Copyright 2020 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_COMPRESS_TRANSFORMS_QUANTIZED_CONVERSIONS_H_
#define IREE_COMPILER_DIALECT_COMPRESS_TRANSFORMS_QUANTIZED_CONVERSIONS_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Compress {

// Populates conversion patterns that expand `quantized` ops to their original
// (presumably high-precision) form, inserting casts/conversions as necessary.
// This should typically be the lowest priority conversion as it will defer
// to emulating any quantization decisions in terms of original (floating
// point) ops.
void populateQuantizedFallbackPatterns(MLIRContext *context,
                                       OwningRewritePatternList &patterns);

}  // namespace Compress
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_COMPRESS_TRANSFORMS_QUANTIZED_CONVERSIONS_H_

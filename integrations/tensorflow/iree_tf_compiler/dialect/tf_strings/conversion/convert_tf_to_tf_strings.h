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

#ifndef IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_TFTOTFSTRINGSS_H_
#define IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_TFTOTFSTRINGSS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTFToTFStringsPass();

// Adds rewrite patterns for lowering tensorflow operations to tf_strings.
void populateTFToTFStringsPatterns(MLIRContext *ctx,
                                   OwningRewritePatternList &patterns);

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TFSTRINGS_TRANSFORMS_TFSTRINGSTOSTRINGS_H_

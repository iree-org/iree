//===- GemminiLegalizeStubs.cpp - Stub implementations for LLVM export ----===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file provides stub implementations for Gemmini LLVM export functions
// to allow the build to succeed. These will be replaced with actual
// implementations when IREE_ENABLE_BUDDY_GEMMINI_LEGALIZE is enabled.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

// Stub implementation - to be replaced when LegalizeForLLVMExport.cpp is fixed
void configureGemminiLegalizeForExportTarget(LLVMConversionTarget &target) {
  // TODO: Implement when IREE_ENABLE_BUDDY_GEMMINI_LEGALIZE is enabled
}

// Stub implementation - to be replaced when LegalizeForLLVMExport.cpp is fixed  
void populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, 
    int64_t dim, int64_t addrLen, int64_t accRows, int64_t bankRows,
    size_t sizeOfElemT, size_t sizeOfAccT) {
  // TODO: Implement when IREE_ENABLE_BUDDY_GEMMINI_LEGALIZE is enabled
}

} // namespace mlir

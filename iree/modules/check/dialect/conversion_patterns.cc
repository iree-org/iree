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

#include "iree/modules/check/dialect/conversion_patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/modules/check/dialect/check_ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Check {

void populateCheckToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                               OwningRewritePatternList &patterns,
                               TypeConverter &typeConverter) {
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectTrueOp>>(
      context, importSymbols, typeConverter, "check.expect_true");
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectFalseOp>>(
      context, importSymbols, typeConverter, "check.expect_false");
  patterns.insert<VMImportOpConversion<IREE::Check::ExpectAllTrueOp>>(
      context, importSymbols, typeConverter, "check.expect_all_true");
}

void populateCheckToHALPatterns(MLIRContext *context,
                                OwningRewritePatternList &patterns,
                                TypeConverter &typeConverter) {
  // The same op handles both tensors and buffer views.
  patterns.insert<HALOpConversion<IREE::Check::ExpectAllTrueOp,
                                  IREE::Check::ExpectAllTrueOp>>(context,
                                                                 typeConverter);
}

}  // namespace Check
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

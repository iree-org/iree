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

#include "iree/samples/custom_modules/dialect/conversion_patterns.h"

#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/samples/custom_modules/dialect/custom_dialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Custom {

void populateCustomToVMPatterns(MLIRContext *context,
                                SymbolTable &importSymbols,
                                OwningRewritePatternList &patterns,
                                TypeConverter &typeConverter) {
  patterns.insert<VMImportOpConversion<IREE::Custom::PrintOp>>(
      context, importSymbols, typeConverter, "custom.print");
  patterns.insert<VMImportOpConversion<IREE::Custom::ReverseOp>>(
      context, importSymbols, typeConverter, "custom.reverse");
  patterns.insert<VMImportOpConversion<IREE::Custom::GetUniqueMessageOp>>(
      context, importSymbols, typeConverter, "custom.get_unique_message");
}

}  // namespace Custom
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

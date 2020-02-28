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

#include "iree/compiler/Dialect/Modules/TensorList/Conversion/ConvertHALToVM.h"

#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace TensorList {

void populateTensorListToVMPatterns(MLIRContext *context,
                                    SymbolTable &importSymbols,
                                    OwningRewritePatternList &patterns,
                                    TypeConverter &typeConverter) {
  patterns.insert<VMImportOpConversion<IREE::TensorList::Reserve>>(
      context, importSymbols, typeConverter, "tensorlist.reserve");
  patterns.insert<VMImportOpConversion<IREE::TensorList::GetItem>>(
      context, importSymbols, typeConverter, "tensorlist.get_item");
  patterns.insert<VMImportOpConversion<IREE::TensorList::SetItem>>(
      context, importSymbols, typeConverter, "tensorlist.set_item");
  patterns.insert<VMImportOpConversion<IREE::TensorList::FromTensor>>(
      context, importSymbols, typeConverter, "tensorlist.from_tensor");
  patterns.insert<VMImportOpConversion<IREE::TensorList::Stack>>(
      context, importSymbols, typeConverter, "tensorlist.stack");
}

}  // namespace TensorList
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

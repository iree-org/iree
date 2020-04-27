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

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateHALSemaphoreToVMPatterns(MLIRContext *context,
                                      SymbolTable &importSymbols,
                                      TypeConverter &typeConverter,
                                      OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreCreateOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreQueryOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.query");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreSignalOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.signal");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreFailOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.fail");
  patterns.insert<VMImportOpConversion<IREE::HAL::SemaphoreAwaitOp>>(
      context, importSymbols, typeConverter, "hal.semaphore.await");
}

}  // namespace iree_compiler
}  // namespace mlir

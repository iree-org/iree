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
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateHALExperimentalToVMPatterns(MLIRContext *context,
                                         SymbolTable &importSymbols,
                                         TypeConverter &typeConverter,
                                         OwningRewritePatternList &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::ExSharedDeviceOp>>(
      context, importSymbols, typeConverter, "_hal.ex.shared_device");
  patterns.insert<VMImportOpConversion<IREE::HAL::ExPushBindingOp>>(
      context, importSymbols, typeConverter, "_hal.ex.push_binding");
  patterns.insert<
      VMImportOpConversion<IREE::HAL::ExExecutableDescriptorSetLayoutOp>>(
      context, importSymbols, typeConverter,
      "_hal.ex.executable_descriptor_set_layout");
  patterns.insert<VMImportOpConversion<IREE::HAL::ExDeferReleaseOp>>(
      context, importSymbols, typeConverter, "_hal.ex.defer_release");
  patterns.insert<VMImportOpConversion<IREE::HAL::ExSubmitAndWaitOp>>(
      context, importSymbols, typeConverter, "_hal.ex.submit_and_wait");
}

}  // namespace iree_compiler
}  // namespace mlir

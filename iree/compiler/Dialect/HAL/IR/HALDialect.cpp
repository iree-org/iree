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

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/ConvertHALToVM.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/HAL/hal.imports.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

static DialectRegistration<HALDialect> hal_dialect;

class HALToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningModuleRef getVMImportModule() const override {
    return mlir::parseSourceString(
        StringRef(hal_imports_create()->data, hal_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, OwningRewritePatternList &patterns,
      TypeConverter &typeConverter) const override {
    populateHALToVMPatterns(getDialect()->getContext(), importSymbols, patterns,
                            typeConverter);
  }
};

}  // namespace

HALDialect::HALDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addInterfaces<HALToVMConversionInterface>();

  addTypes<AllocatorType, BufferType, CommandBufferType, DescriptorSetType,
           DescriptorSetLayoutType, DeviceType, EventType, ExecutableType,
           ExecutableCacheType, FenceType, RingBufferType, SemaphoreType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/HAL/IR/HALOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Type printing and parsing
//===----------------------------------------------------------------------===//

Type HALDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (parser.parseKeyword(&typeName)) return Type();
  auto type =
      llvm::StringSwitch<Type>(typeName)
          .Case("allocator", AllocatorType::get(getContext()))
          .Case("buffer", BufferType::get(getContext()))
          .Case("command_buffer", CommandBufferType::get(getContext()))
          .Case("descriptor_set", DescriptorSetType::get(getContext()))
          .Case("descriptor_set_layout",
                DescriptorSetLayoutType::get(getContext()))
          .Case("device", DeviceType::get(getContext()))
          .Case("event", EventType::get(getContext()))
          .Case("executable", ExecutableType::get(getContext()))
          .Case("executable_cache", ExecutableCacheType::get(getContext()))
          .Case("fence", FenceType::get(getContext()))
          .Case("ring_buffer", RingBufferType::get(getContext()))
          .Case("semaphore", SemaphoreType::get(getContext()))
          .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown HAL type: " << typeName;
  }
  return type;
}

void HALDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<AllocatorType>()) {
    p << "allocator";
  } else if (type.isa<BufferType>()) {
    p << "buffer";
  } else if (type.isa<CommandBufferType>()) {
    p << "command_buffer";
  } else if (type.isa<DescriptorSetType>()) {
    p << "descriptor_set";
  } else if (type.isa<DescriptorSetLayoutType>()) {
    p << "descriptor_set_layout";
  } else if (type.isa<DeviceType>()) {
    p << "device";
  } else if (type.isa<EventType>()) {
    p << "event";
  } else if (type.isa<ExecutableType>()) {
    p << "executable";
  } else if (type.isa<ExecutableCacheType>()) {
    p << "executable_cache";
  } else if (type.isa<FenceType>()) {
    p << "fence";
  } else if (type.isa<RingBufferType>()) {
    p << "ring_buffer";
  } else if (type.isa<SemaphoreType>()) {
    p << "semaphore";
  } else {
    llvm_unreachable("unknown HAL type");
  }
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONSTEVAL_RUNTIME_H_
#define IREE_COMPILER_CONSTEVAL_RUNTIME_H_

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::iree_compiler::ConstEval {

// Abstract base class for a compiled binary.
class CompiledBinary {
public:
  using ResultsCallback = std::function<LogicalResult(iree_vm_list_t *outputs)>;
  virtual ~CompiledBinary();

  iree_hal_allocator_t *getAllocator() {
    return iree_hal_device_allocator(device.get());
  }

protected:
  CompiledBinary();
  void initialize(void *data, size_t length);
  // The base class does not clean up initialized state. This must be done
  // explicitly by subclasses, ensuring that any backing images remain valid
  // through the call to deinitialize().
  void deinitialize();
  TypedAttr convertVariantToAttribute(Location loc, iree_vm_variant_t &variant,
                                      Type mlirType);

  iree::vm::ref<iree_hal_device_t> device;
  iree::vm::ref<iree_vm_module_t> hal_module;
  iree::vm::ref<iree_vm_module_t> main_module;
  iree::vm::ref<iree_vm_context_t> context;

  friend class FunctionCall;
};

class FunctionCall {
public:
  FunctionCall(CompiledBinary &binary, iree_host_size_t argCapacity,
               iree_host_size_t resultCapacity);

  LogicalResult addArgument(Location loc, Attribute attr);
  LogicalResult invoke(Location loc, StringRef name);
  LogicalResult getResultAsAttr(Location loc, size_t index, Type mlirType,
                                TypedAttr &outAttr);

private:
  FailureOr<iree::vm::ref<iree_hal_buffer_t>> importSerializableAttr(
      Location loc, IREE::Util::SerializableAttrInterface serializableAttr);
  LogicalResult
  addBufferArgumentAttr(Location loc,
                        IREE::Util::SerializableAttrInterface serializableAttr);
  LogicalResult addBufferViewArgumentAttr(
      Location loc, ShapedType shapedType,
      IREE::Util::SerializableAttrInterface serializableAttr);

  CompiledBinary binary;
  iree::vm::ref<iree_vm_list_t> inputs;
  iree::vm::ref<iree_vm_list_t> outputs;
};

// An in-memory compiled binary and accessors for working with it.
class InMemoryCompiledBinary : public CompiledBinary {
public:
  LogicalResult translateFromModule(mlir::ModuleOp moduleOp);
  ~InMemoryCompiledBinary() override;

private:
  std::string binary;
};

// Simple wrapper around IREE runtime library sufficient for loading and
// executing simple programs.
class Runtime {
public:
  static Runtime &getInstance();

  iree_hal_driver_registry_t *registry = nullptr;
  iree::vm::ref<iree_vm_instance_t> instance;

private:
  Runtime();
  ~Runtime();
};

} // namespace mlir::iree_compiler::ConstEval

#endif // IREE_COMPILER_CONSTEVAL_RUNTIME_H_

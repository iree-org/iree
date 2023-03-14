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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {

class ModuleOp;

namespace iree_compiler {
namespace ConstEval {

// Abstract base class for a compiled binary.
class CompiledBinary {
 public:
  using ResultsCallback = std::function<LogicalResult(iree_vm_list_t* outputs)>;
  virtual ~CompiledBinary();

  // Invokes a nullary function.
  LogicalResult invokeNullary(Location loc, StringRef name,
                              ResultsCallback callback);

  // Invokes a nullary function and returns its (presumed single) single result
  // as an Attribute.
  Attribute invokeNullaryAsAttribute(Location loc, StringRef name);

  // Whether the given type is supported in *AsAttribute methods.
  static bool isSupportedResultType(Type type);

 protected:
  CompiledBinary();
  void initialize(void* data, size_t length);
  // The base class does not clean up initialized state. This must be done
  // explicitly by subclasses, ensuring that any backing images remain valid
  // through the call to deinitialize().
  void deinitialize();
  Attribute convertVariantToAttribute(Location loc, iree_vm_variant_t& variant);

  iree::vm::ref<iree_hal_device_t> device;
  iree::vm::ref<iree_vm_module_t> hal_module;
  iree::vm::ref<iree_vm_module_t> main_module;
  iree::vm::ref<iree_vm_context_t> context;
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
  static Runtime& getInstance();

  iree_hal_driver_registry_t* registry = nullptr;
  iree::vm::ref<iree_vm_instance_t> instance;

 private:
  Runtime();
  ~Runtime();
};

}  // namespace ConstEval
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONSTEVAL_RUNTIME_H_

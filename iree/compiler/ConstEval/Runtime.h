// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONSTEVAL_RUNTIME_H_
#define IREE_COMPILER_CONSTEVAL_PASSES_H_

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/hal/driver_registry.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"
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
  Attribute invokeNullaryAsElements(Location loc, StringRef name);

 protected:
  CompiledBinary();
  void initialize(void* data, size_t length);
  Attribute convertVariantToAttribute(Location loc, iree_vm_variant_t& variant);

  iree_vm_instance_t* instance = nullptr;
  iree_vm_context_t* context = nullptr;
  iree_vm_module_t* main_module = nullptr;
};

// An in-memory compiled binary and accessors for working with it.
class InMemoryCompiledBinary : public CompiledBinary {
 public:
  LogicalResult translateFromModule(mlir::ModuleOp moduleOp);

 private:
  std::string binary;
};

// Simple wrapper around IREE runtime library sufficient for loading and
// executing simple programs.
class Runtime {
 public:
  static Runtime& getInstance();

  iree_hal_driver_registry_t* registry = nullptr;
  iree_hal_device_t* device = nullptr;
  iree_vm_module_t* hal_module = nullptr;

 private:
  Runtime();
};

}  // namespace ConstEval
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONSTEVAL_RUNTIME_H_

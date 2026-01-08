// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_UTILS_H_
#define IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_UTILS_H_

#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::IREE::Stream {

//===----------------------------------------------------------------------===//
// Dialect Interface Utilities
//===----------------------------------------------------------------------===//

/// Returns a stably sorted list of dialect interfaces of T for all dialects
/// used within the given module.
template <typename T>
SmallVector<const T *> gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
  SmallPtrSet<const T *, 4> resultSet;
  for (auto dialect : moduleOp.getContext()->getLoadedDialects()) {
    auto *dialectInterface = dialect->getRegisteredInterface<T>();
    if (!dialectInterface)
      continue;
    resultSet.insert(dialectInterface);
  }

  // NOTE: to ensure deterministic output we sort the result so that imports are
  // always added in a consistent order.
  auto results = llvm::to_vector_of<const T *>(resultSet);
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}

//===----------------------------------------------------------------------===//
// Executable Encoding Utilities
//===----------------------------------------------------------------------===//

/// Returns the operands encodings and result encodings from the `dispatchOp` in
/// |operands| + |results| order, i.e., it returns the stripped concatenated
/// operand encodings and result encodings. If a result is tied to an operand,
/// the result encoding is skipped because it shares the same binding with the
/// tied operand.
SmallVector<Attribute> getBindingLayoutAttrs(TensorDispatchOp dispatchOp);

/// Returns true iff all the entry points are recognized by the pass:
///   - The corresponding executable is a stream.executable op.
///   - The function arguments, where the types are !stream.binding_type, are
///     only used by stream.binding.subspan ops. Furthermore, the result type of
///     subspan ops have to implement IREE::Encoding::EncodingTypeInterface.
bool recognizeDispatchEntryPoints(ModuleOp moduleOp, SymbolTable &symbolTable,
                                  TensorDispatchOp dispatchOp);

/// Updates the bindings of function arguments with encoding layouts. It only
/// updates the uses when the argument type is stream.binding_type. The bindings
/// are only used by binding subspan ops that return whatever types. Today they
/// are mostly flow tensor type. If the type implements
/// IREE::Encoding::EncodingTypeInterface type interface, the method uses the
/// interface methods to compute the type that has updated encodings (i.e.,
/// encodings with layouts) and updates the type.
LogicalResult updateBindingEncodings(FunctionOpInterface funcOp,
                                     ArrayRef<Attribute> bindingLayoutTypeAttrs);

/// Duplicates stream.executables based on the operand encodings and result
/// encodings of stream.tensor.dispatch ops. Some executables can be launched by
/// different devices. It can produce wrong codegen artifacts when bindings
/// types are encoded (i.e., the tensor type has an encoding attribute). Because
/// they can result in different layouts, especially when multi-device is
/// involved. E.g., say that device_a and device_b interpret a tensor type with
/// encodings in different layouts, and there is an executable that can be
/// launch with resources from either device_a or device_b. It is confusing what
/// the input layouts for the executable because there are two possibilities. In
/// this case, we have to duplicate the executable with updated encoding, and
/// modify the dispatch to launch proper executable based on resolved encoding
/// layouts.
LogicalResult
duplicateExecutablesPerLayoutVariant(ModuleOp moduleOp,
                                     SymbolTable &symbolTable,
                                     ArrayRef<TensorDispatchOp> candidates);

} // namespace mlir::iree_compiler::IREE::Stream

#endif // IREE_COMPILER_DIALECT_STREAM_TRANSFORMS_UTILS_H_

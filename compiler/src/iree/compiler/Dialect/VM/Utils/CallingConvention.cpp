// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Utils/CallingConvention.h"

#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

// Encodes a type (or a tuple of nested types) to a calling convention string.
//
// Examples:
//  i32              -> i
//  !vm.ref<...>     -> r
//  tuple<i32, i64>  -> iI
LogicalResult encodeCallingConventionType(Operation *op, Type type,
                                          SmallVectorImpl<char> &s) {
  if (auto refPtrType = type.dyn_cast<IREE::VM::RefType>()) {
    s.push_back('r');
    return success();
  } else if (auto integerType = type.dyn_cast<IntegerType>()) {
    switch (integerType.getIntOrFloatBitWidth()) {
      default:
      case 32:
        s.push_back('i');
        return success();
      case 64:
        s.push_back('I');
        return success();
    }
  } else if (auto floatType = type.dyn_cast<FloatType>()) {
    switch (floatType.getIntOrFloatBitWidth()) {
      default:
      case 32:
        s.push_back('f');
        return success();
      case 64:
        s.push_back('F');
        return success();
    }
  } else if (auto tupleType = type.dyn_cast<TupleType>()) {
    // Flatten tuple (so tuple<i32, i64> -> `...iI...`).
    SmallVector<Type, 4> flattenedTypes;
    tupleType.getFlattenedTypes(flattenedTypes);
    for (auto elementType : flattenedTypes) {
      if (failed(encodeCallingConventionType(op, elementType, s))) {
        return op->emitError()
               << "unsupported external calling convention tuple element type "
               << elementType;
      }
    }
    return success();
  }
  return op->emitError() << "unsupported external calling convention type "
                         << type;
}

LogicalResult encodeVariadicCallingConventionType(Operation *op, Type type,
                                                  SmallVectorImpl<char> &s) {
  s.push_back('C');
  auto result = encodeCallingConventionType(op, type, s);
  s.push_back('D');
  return result;
}

std::optional<std::string> makeImportCallingConventionString(
    IREE::VM::ImportOp importOp) {
  auto functionType = importOp.getFunctionType();
  if (functionType.getNumInputs() == 0 && functionType.getNumResults() == 0) {
    return std::string("0v_v");  // Valid but empty.
  }

  SmallVector<char, 8> s = {'0'};
  if (functionType.getNumInputs() > 0) {
    for (int i = 0; i < functionType.getNumInputs(); ++i) {
      if (importOp.isFuncArgumentVariadic(i)) {
        if (failed(encodeVariadicCallingConventionType(
                importOp, functionType.getInput(i), s))) {
          return std::nullopt;
        }
      } else {
        if (failed(encodeCallingConventionType(importOp,
                                               functionType.getInput(i), s))) {
          return std::nullopt;
        }
      }
    }
  } else {
    s.push_back('v');
  }
  s.push_back('_');
  if (functionType.getNumResults() > 0) {
    for (int i = 0; i < functionType.getNumResults(); ++i) {
      if (failed(encodeCallingConventionType(importOp,
                                             functionType.getResult(i), s))) {
        return std::nullopt;
      }
    }
  } else {
    s.push_back('v');
  }
  return std::string(s.data(), s.size());
}

std::optional<std::string> makeCallingConventionString(
    IREE::VM::FuncOp funcOp) {
  auto functionType = funcOp.getFunctionType();
  if (functionType.getNumInputs() == 0 && functionType.getNumResults() == 0) {
    return std::string("0v_v");  // Valid but empty.
  }

  SmallVector<char, 8> s = {'0'};
  if (functionType.getNumInputs() > 0) {
    for (int i = 0; i < functionType.getNumInputs(); ++i) {
      if (failed(encodeCallingConventionType(funcOp, functionType.getInput(i),
                                             s))) {
        return std::nullopt;
      }
    }
  } else {
    s.push_back('v');
  }
  s.push_back('_');
  if (functionType.getNumResults() > 0) {
    for (int i = 0; i < functionType.getNumResults(); ++i) {
      if (failed(encodeCallingConventionType(funcOp, functionType.getResult(i),
                                             s))) {
        return std::nullopt;
      }
    }
  } else {
    s.push_back('v');
  }
  return std::string(s.data(), s.size());
}

}  // namespace VM
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

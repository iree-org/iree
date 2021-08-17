// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_IR_IREEOPS_H_
#define IREE_COMPILER_DIALECT_IREE_IR_IREEOPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilOps.h.inc"  // IWYU pragma: export

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                  StringAttr &symVisibilityAttr);
void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                           StringAttr symVisibilityAttr);

//===----------------------------------------------------------------------===//
// custom<TypeOrAttr>($type, $attr)
//===----------------------------------------------------------------------===//
// some.op custom<TypeOrAttr>($type, $attr)
// ->
// some.op : i32
// some.op = 42 : i32
// some.op : i32 = 42 : index

ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                            Attribute &attr);
void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                     Attribute attr);

//===----------------------------------------------------------------------===//
// custom<SizeAwareType>
//===----------------------------------------------------------------------===//
// type{%size}

ParseResult parseSizeAwareType(OpAsmParser &parser, Type &type,
                               OpAsmParser::OperandType &size);
void printSizeAwareType(OpAsmPrinter &p, Operation *op, Type type, Value size);

//===----------------------------------------------------------------------===//
// custom<SizeAwareTypeList>
//===----------------------------------------------------------------------===//
// (type{%size0}, type, type{%size1})

ParseResult parseSizeAwareTypeList(
    OpAsmParser &parser, SmallVectorImpl<Type> &types,
    SmallVectorImpl<OpAsmParser::OperandType> &sizes);
void printSizeAwareTypeList(OpAsmPrinter &p, Operation *op, TypeRange types,
                            OperandRange sizes);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_IR_IREEOPS_H_

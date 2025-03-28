// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/AsmParser/AsmParserState.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::iree_compiler {

ParseResult
parseIndexVecs(OpAsmParser &parser,
               SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indexVecs,
               SmallVectorImpl<Type> &indexVecTypes,
               DenseI64ArrayAttr &indexed);
void printIndexVecs(OpAsmPrinter &printer, Operation *op,
                    OperandRange indexVecs, TypeRange indexVecTypes,
                    DenseI64ArrayAttr indexed);

} // namespace mlir::iree_compiler

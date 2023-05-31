// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

void populateHALChannelToVMPatterns(MLIRContext *context,
                                    SymbolTable &importSymbols,
                                    TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  patterns.insert<VMImportOpConversion<IREE::HAL::ChannelCreateOp>>(
      context, importSymbols, typeConverter, "hal.channel.create");
  patterns.insert<VMImportOpConversion<IREE::HAL::ChannelSplitOp>>(
      context, importSymbols, typeConverter, "hal.channel.split");
  patterns.insert<VMImportOpConversion<IREE::HAL::ChannelRankAndCountOp>>(
      context, importSymbols, typeConverter, "hal.channel.rank_and_count");
}

}  // namespace iree_compiler
}  // namespace mlir

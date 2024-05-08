// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Conversion/HALToVM/Patterns.h"

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler {

extern void populateHALAllocatorToVMPatterns(MLIRContext *context,
                                             SymbolTable &importSymbols,
                                             TypeConverter &typeConverter,
                                             RewritePatternSet &patterns);
extern void populateHALBufferToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
extern void populateHALBufferViewToVMPatterns(MLIRContext *context,
                                              SymbolTable &importSymbols,
                                              TypeConverter &typeConverter,
                                              RewritePatternSet &patterns);
extern void populateHALChannelToVMPatterns(MLIRContext *context,
                                           SymbolTable &importSymbols,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);
extern void populateHALCommandBufferToVMPatterns(MLIRContext *context,
                                                 SymbolTable &importSymbols,
                                                 TypeConverter &typeConverter,
                                                 RewritePatternSet &patterns);
extern void populateHALDeviceToVMPatterns(MLIRContext *context,
                                          SymbolTable &importSymbols,
                                          TypeConverter &typeConverter,
                                          RewritePatternSet &patterns);
extern void populateHALDevicesToVMPatterns(MLIRContext *context,
                                           SymbolTable &importSymbols,
                                           TypeConverter &typeConverter,
                                           RewritePatternSet &patterns);
extern void populateHALExecutableToVMPatterns(MLIRContext *context,
                                              SymbolTable &importSymbols,
                                              TypeConverter &typeConverter,
                                              RewritePatternSet &patterns);
extern void populateHALExperimentalToVMPatterns(MLIRContext *context,
                                                SymbolTable &importSymbols,
                                                TypeConverter &typeConverter,
                                                RewritePatternSet &patterns);
extern void populateHALFenceToVMPatterns(MLIRContext *context,
                                         SymbolTable &importSymbols,
                                         TypeConverter &typeConverter,
                                         RewritePatternSet &patterns);

void populateHALToVMPatterns(MLIRContext *context, SymbolTable &importSymbols,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter) {
  populateHALAllocatorToVMPatterns(context, importSymbols, typeConverter,
                                   patterns);
  populateHALBufferToVMPatterns(context, importSymbols, typeConverter,
                                patterns);
  populateHALBufferViewToVMPatterns(context, importSymbols, typeConverter,
                                    patterns);
  populateHALChannelToVMPatterns(context, importSymbols, typeConverter,
                                 patterns);
  populateHALCommandBufferToVMPatterns(context, importSymbols, typeConverter,
                                       patterns);
  populateHALDeviceToVMPatterns(context, importSymbols, typeConverter,
                                patterns);
  populateHALDevicesToVMPatterns(context, importSymbols, typeConverter,
                                 patterns);
  populateHALExecutableToVMPatterns(context, importSymbols, typeConverter,
                                    patterns);
  populateHALExperimentalToVMPatterns(context, importSymbols, typeConverter,
                                      patterns);
  populateHALFenceToVMPatterns(context, importSymbols, typeConverter, patterns);
}

} // namespace mlir::iree_compiler

// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/TransformStrategies/GPU/MappingInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

void mlir::iree_compiler::gpu::MappingInfo::print(llvm::raw_ostream &os) const {
  os << "MappingInfo{";
  os << "vectorSize: " << ((vectorSize.has_value()) ? vectorSize.value() : 0);
  llvm::interleaveComma(numThreads, os << ", numThreads: {");
  llvm::interleaveComma(tileSizes, os << "}, tileSizes: {");
  llvm::interleaveComma(threadMapping, os << "}, threadMapping: {");
  os << "}}";
}

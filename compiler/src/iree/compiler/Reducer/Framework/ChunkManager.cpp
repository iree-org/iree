// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Reducer/Framework/ChunkManager.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::iree_compiler;

bool Chunk::contains(unsigned index) const {
  return index >= begin && index < getEnd();
}

bool mlir::iree_compiler::operator==(const Chunk &C1, const Chunk &C2) {
  return C1.getBegin() == C2.getBegin() && C1.getEnd() == C2.getEnd();
}

bool mlir::iree_compiler::operator!=(const Chunk &C1, const Chunk &C2) {
  return !(C1 == C2);
}

bool mlir::iree_compiler::operator<(const Chunk &C1, const Chunk &C2) {
  return std::make_pair(C1.getBegin(), C1.getEnd()) <
         std::make_pair(C2.getBegin(), C2.getEnd());
}

void Chunk::print(raw_ostream &os) const {
  os << "[" << begin << ", " << end << ")";
}

void Chunk::dump() const {
  print(llvm::errs());
  llvm::errs() << "\n";
}

bool ChunkManager::shouldFeatureBeKept() {
  if (chunksToKeep.empty()) {
    featureIndex++;
    return false;
  }

  // Does the current chunk contain such a feature?
  bool shouldKeep = chunksToKeep.front().contains(featureIndex);

  // If this is the last feature in this chunk, move to the next chunk.
  if (chunksToKeep.front().getEnd() == featureIndex) {
    chunksToKeep = chunksToKeep.drop_front();
  }

  featureIndex++;

  return shouldKeep;
}

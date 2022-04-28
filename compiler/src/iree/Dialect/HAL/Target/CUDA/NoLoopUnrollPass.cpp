// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "iree/compiler/Dialect/HAL/Target/CUDA/LLVMPasses.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

#define DEBUG_TYPE "iree-dialect-hal-cuda-llvm-nounroll"

namespace llvm {
void initializeNoLoopUnrollPass(PassRegistry &Registry);
}

namespace {
/// Pass that mark all loops with llvm.loop.unroll.disable metadata.
class NoLoopUnroll : public LoopPass {
 public:
  static char ID;
  NoLoopUnroll() : LoopPass(ID) {
    initializeNoLoopUnrollPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    L->setLoopAlreadyUnrolled();
    return true;
  }
  StringRef getPassName() const override { return "Set Nounroll pass"; }
};

}  // namespace

char NoLoopUnroll::ID = 0;

INITIALIZE_PASS_BEGIN(NoLoopUnroll, DEBUG_TYPE, "Set Nounroll", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(NoLoopUnroll, DEBUG_TYPE, "Set Nounroll", false, false)

Pass *llvm::createSetNoUnrollPass() { return new NoLoopUnroll(); }

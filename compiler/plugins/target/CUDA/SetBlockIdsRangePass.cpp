// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./LLVMPasses.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

// This is a workaround until nvvm-intr-range gets re-enabled by default in
// NVPTX backend. This also allows us to potentially special more the kernel by
// setting more fine grain ranges based on static dispatch size.
using namespace llvm;

#define DEBUG_TYPE "iree-dialect-hal-cuda-llvm-set-block-ids-range"

// Adds the passed-in [Low,High) range information as metadata to the
// passed-in call instruction.
static bool addRangeMetadata(uint64_t Low, uint64_t High, CallInst *C) {
  // This call already has range metadata, nothing to do.
  if (C->getMetadata(LLVMContext::MD_range)) return false;

  LLVMContext &Context = C->getParent()->getContext();
  IntegerType *Int32Ty = Type::getInt32Ty(Context);
  Metadata *LowAndHigh[] = {
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, Low)),
      ConstantAsMetadata::get(ConstantInt::get(Int32Ty, High))};
  C->setMetadata(LLVMContext::MD_range, MDNode::get(Context, LowAndHigh));
  return true;
}

static bool runOnFunction(Function &F,
                          const std::array<int32_t, 3> &maxWorkgroupSize) {
  bool Changed = false;
  // We could use the number of block dispatched if it is known at compile time
  // however this would prevent re-using kernel re-use. For now just use the API
  // limit.
  unsigned MaxGridSizeX = 0x7fffffff;
  unsigned MaxGridSizeY = 0xffff;
  unsigned MaxGridSizeZ = 0xffff;
  for (Instruction &I : instructions(F)) {
    CallInst *Call = dyn_cast<CallInst>(&I);
    if (!Call) continue;
    Function *Callee = Call->getCalledFunction();
    if (!Callee) continue;
    switch (Callee->getIntrinsicID()) {
      // Index within block
      case Intrinsic::nvvm_read_ptx_sreg_tid_x:
        Changed |= addRangeMetadata(0, maxWorkgroupSize[0], Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_tid_y:
        Changed |= addRangeMetadata(0, maxWorkgroupSize[1], Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_tid_z:
        Changed |= addRangeMetadata(0, maxWorkgroupSize[2], Call);
        break;

      // Block size
      case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
        Changed |= addRangeMetadata(1, maxWorkgroupSize[0] + 1, Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
        Changed |= addRangeMetadata(1, maxWorkgroupSize[1] + 1, Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
        Changed |= addRangeMetadata(1, maxWorkgroupSize[2] + 1, Call);
        break;

      // Index within grid
      case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
        Changed |= addRangeMetadata(0, MaxGridSizeX, Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
        Changed |= addRangeMetadata(0, MaxGridSizeY, Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
        Changed |= addRangeMetadata(0, MaxGridSizeZ, Call);
        break;

      // Grid size
      case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
        Changed |= addRangeMetadata(1, MaxGridSizeX + 1, Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
        Changed |= addRangeMetadata(1, MaxGridSizeY + 1, Call);
        break;
      case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
        Changed |= addRangeMetadata(1, MaxGridSizeZ + 1, Call);
        break;
    }
  }
  return Changed;
}

PreservedAnalyses SetBlockIdsRangePass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  return runOnFunction(F, maxWorkgroupSize) ? PreservedAnalyses::none()
                                            : PreservedAnalyses::all();
}

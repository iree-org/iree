// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#define DEBUG_TYPE "iree-rocdl-add-lds-alias-scopes"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLADDLDSALIASSCOPESPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

/// Trace an LDS pointer back through GEP and extractvalue operations to find
/// the root value representing the underlying LDS allocation. After
/// AMDGPU-to-ROCDL conversion, memref descriptors are lowered to LLVM structs,
/// and LDS pointers are obtained via extractvalue (to get the base pointer from
/// the descriptor) followed by GEP (to compute offsets). This function walks
/// backwards through that chain so that pointers derived from the same
/// allocation map to the same root value.
static Value traceToLDSBase(Value ldsPtr) {
  Value current = ldsPtr;
  bool changed = true;
  while (changed) {
    changed = false;
    if (auto gepOp = current.getDefiningOp<LLVM::GEPOp>()) {
      current = gepOp.getBase();
      changed = true;
    } else if (auto extractOp = current.getDefiningOp<LLVM::ExtractValueOp>()) {
      current = extractOp.getContainer();
      changed = true;
    }
  }
  return current;
}

/// Adds alias scope metadata to ROCDL::LoadToLDSOp operations.
///
/// This pass enables the LLVM backend to distinguish between different LDS
/// buffers used in multi-buffered pipelining. Without alias scopes, the
/// backend conservatively inserts s_waitcnt vmcnt(0) before every LDS access
/// that might alias with a prior LoadToLDS operation.
///
/// With alias scopes, the backend understands that:
/// - LoadToLDS ops to buffer 0 don't alias with ops to buffer 1
/// - ds_read from buffer 0 doesn't conflict with LoadToLDS to buffer 1
///
/// This allows the backend to eliminate redundant vmcnt(0) instructions,
/// enabling better overlap between global loads and LDS reads.
struct ROCDLAddLDSAliasScopesPass final
    : impl::ROCDLAddLDSAliasScopesPassBase<ROCDLAddLDSAliasScopesPass> {

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    // Create a single alias scope domain for LDS DMA operations.
    auto domainAttr = LLVM::AliasScopeDomainAttr::get(
        ctx, StringAttr::get(ctx, "lds_dma_domain"));

    DenseMap<Value, LLVM::AliasScopeAttr> baseToScope;
    SmallVector<LLVM::AliasScopeAttr> allScopes;
    // First pass: create unique alias scopes for each distinct LDS allocation.
    m.walk([&](ROCDL::LoadToLDSOp loadOp) {
      Value base = traceToLDSBase(loadOp.getLdsPtr());
      if (!baseToScope.contains(base)) {
        std::string scopeName =
            "lds_buffer_" + std::to_string(baseToScope.size());
        auto scopeAttr = LLVM::AliasScopeAttr::get(
            domainAttr, StringAttr::get(ctx, scopeName));
        baseToScope[base] = scopeAttr;
        allScopes.push_back(scopeAttr);
      }
    });

    // Only add metadata when there are multiple distinct LDS buffers.
    // With a single buffer, alias scopes provide no disambiguation benefit.
    if (allScopes.size() <= 1) {
      return;
    }

    LDBG() << "Adding alias scopes to " << allScopes.size()
           << " distinct LDS buffers";

    // Second pass: attach alias_scopes and noalias_scopes to each LoadToLDSOp.
    m.walk([&](ROCDL::LoadToLDSOp loadOp) {
      Value base = traceToLDSBase(loadOp.getLdsPtr());
      LLVM::AliasScopeAttr scopeAttr = baseToScope.lookup(base);

      loadOp.setAliasScopesAttr(ArrayAttr::get(ctx, {scopeAttr}));
      SmallVector<Attribute> noaliasScopes;
      for (LLVM::AliasScopeAttr otherScope : allScopes) {
        if (otherScope != scopeAttr) {
          noaliasScopes.push_back(otherScope);
        }
      }
      loadOp.setNoaliasScopesAttr(ArrayAttr::get(ctx, noaliasScopes));

      LDBG() << "Added alias scope to LoadToLDSOp: " << loadOp;
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler

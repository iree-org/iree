// Copyright 2025 The IREE Authors
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

/// Adds alias scope metadata to ROCDL::LoadToLDSOp operations.
///
/// This pass enables the LLVM backend to distinguish between different LDS
/// buffers used in double-buffered pipelining. Without alias scopes, the
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

    // Create a single alias scope domain for LDS DMA operations
    auto domainAttr = LLVM::AliasScopeDomainAttr::get(
        ctx, StringAttr::get(ctx, "lds_dma_domain"));

    // Map from base LDS pointer to its alias scope
    DenseMap<Value, LLVM::AliasScopeAttr> ptrToScope;
    SmallVector<LLVM::AliasScopeAttr> allScopes;

    // First pass: create unique alias scopes for each distinct LDS base pointer
    m.walk([&](ROCDL::LoadToLDSOp loadOp) {
      Value ldsPtr = loadOp.getLdsPtr();

      // Trace back to find the base allocation
      Value basePtr = ldsPtr;
      while (auto gepOp = basePtr.getDefiningOp<LLVM::GEPOp>()) {
        basePtr = gepOp.getBase();
      }

      if (!ptrToScope.count(basePtr)) {
        std::string scopeName =
            "lds_buffer_" + std::to_string(ptrToScope.size());
        auto scopeAttr = LLVM::AliasScopeAttr::get(
            domainAttr, StringAttr::get(ctx, scopeName));
        ptrToScope[basePtr] = scopeAttr;
        allScopes.push_back(scopeAttr);
      }
    });

    if (allScopes.empty())
      return;

    LDBG() << "Adding alias scopes to " << allScopes.size()
           << " distinct LDS buffers";

    // Second pass: attach alias scopes and noalias scopes to each LoadToLDSOp
    m.walk([&](ROCDL::LoadToLDSOp loadOp) {
      Value ldsPtr = loadOp.getLdsPtr();

      // Trace back to find the base allocation
      Value basePtr = ldsPtr;
      while (auto gepOp = basePtr.getDefiningOp<LLVM::GEPOp>()) {
        basePtr = gepOp.getBase();
      }

      auto scopeAttr = ptrToScope.lookup(basePtr);
      if (!scopeAttr)
        return;

      // Set alias_scopes to this buffer's scope
      SmallVector<Attribute> aliasScopes = {scopeAttr};
      loadOp.setAliasScopesAttr(ArrayAttr::get(ctx, aliasScopes));

      // Set noalias_scopes to all OTHER buffer scopes
      SmallVector<Attribute> noaliasScopes;
      for (auto otherScope : allScopes) {
        if (otherScope != scopeAttr) {
          noaliasScopes.push_back(otherScope);
        }
      }
      if (!noaliasScopes.empty()) {
        loadOp.setNoaliasScopesAttr(ArrayAttr::get(ctx, noaliasScopes));
      }

      LDBG() << "Added alias scope to LoadToLDSOp: " << loadOp;
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler

// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-specialize-encodings"

#define GEN_PASS_DEF_SPECIALIZEENCODINGSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {
// Returns a stably sorted list of dialect interfaces of T for all dialects used
// within the given module.
template <typename T>
SmallVector<const T *> gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
  SmallPtrSet<const T *, 4> resultSet;
  for (auto dialect : moduleOp.getContext()->getLoadedDialects()) {
    auto *dialectInterface = dialect->getRegisteredInterface<T>();
    if (!dialectInterface)
      continue;
    resultSet.insert(dialectInterface);
  }

  // NOTE: to ensure deterministic output we sort the result so that imports are
  // always added in a consistent order.
  SmallVector<const T *> results = {resultSet.begin(), resultSet.end()};
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}

static SmallVector<Attribute>
getOperandsResourceAffinities(AffinityAnalysis &affinityAnalysis,
                              Stream::AsyncDispatchOp dispatchOp) {
  MLIRContext *ctx = dispatchOp.getContext();
  SmallVector<Attribute> operandAttrs;
  auto emptyArray = ArrayAttr::get(ctx, {});
  for (auto operand : dispatchOp.getResourceOperands()) {
    if (!isa<IREE::Stream::AffinityTypeInterface>(operand.getType())) {
      continue;
    }
    SmallVector<IREE::Stream::AffinityAttr> affinities;
    if (affinityAnalysis.tryLookupResourceAffinity(operand, affinities)) {
      operandAttrs.push_back(
          ArrayAttr::get(ctx, llvm::to_vector_of<Attribute>(affinities)));
    } else {
      operandAttrs.push_back(emptyArray);
    }
  }
  return operandAttrs;
}

static void updateExecutableOpEncodings(
    ModuleOp moduleOp, Stream::ExecutableOp executableOp,
    ArrayRef<Attribute> operandAttrs, AffinityAttr resultAffinity,
    SymbolTable symbolTable,
    std::function<LogicalResult(AffinityAttr, Operation *,
                                SetVector<Attribute> &)>
        resolver) {
  LLVM_DEBUG(llvm::dbgs() << "Update ExecutableOp: "
                          << executableOp.getSymName() << "\n");
  LLVM_DEBUG({
    llvm::dbgs() << "  operand affinities: [";
    llvm::interleaveComma(operandAttrs, llvm::dbgs());
    llvm::dbgs() << "]\n";
  });

  MLIRContext *ctx = executableOp.getContext();
  for (auto exportOp :
       executableOp.getOps<IREE::Stream::ExecutableExportOp>()) {
    exportOp.getSymName();
    auto funcOp = cast<mlir::FunctionOpInterface>(symbolTable.lookupSymbolIn(
        executableOp.getInnerModule(), exportOp.getSymName()));
    Region &region = funcOp.getFunctionBody();
    auto argsAffinities = llvm::map_to_vector(
        operandAttrs, [](Attribute attr) { return cast<ArrayAttr>(attr); });
    auto resAffinityAttr =
        ArrayAttr::get(ctx, {cast<Attribute>(resultAffinity)});
    argsAffinities.resize(region.getNumArguments(), resAffinityAttr);
    int idx = 0;
    for (auto arg : region.getArguments()) {
      if (!isa<IREE::Stream::BindingType>(arg.getType())) {
        continue;
      }
      ArrayRef<Attribute> affinities = argsAffinities[idx++].getValue();
      assert(affinities.size() == 1);
      SetVector<Attribute> resolvedTargets;
      if (failed(resolver(cast<Stream::AffinityAttr>(affinities[0]), moduleOp,
                          resolvedTargets))) {
        LLVM_DEBUG(llvm::dbgs() << "failed on getting target resolvers\n");
        continue;
      }

      for (auto user : arg.getUsers()) {
        // TODO(hanchung): Is it the only case?
        auto subspanOp = cast<IREE::Stream::BindingSubspanOp>(user);
        auto resType =
            dyn_cast<IREE::Flow::DispatchTensorType>(subspanOp.getType());
        if (!resType) {
          continue;
        }
        auto tensorType = dyn_cast<RankedTensorType>(resType.getBoundType());
        if (!tensorType || !tensorType.getEncoding()) {
          continue;
        }
        auto encoding =
            dyn_cast<Encoding::EncodingAttr>(tensorType.getEncoding());

        SmallVector<Attribute> targets(resolvedTargets.begin(),
                                       resolvedTargets.end());
        subspanOp.getResult().setType(
            resType.updateEncoding(encoding.cloneWithTargets(targets)));
      }
    }
  }
}

} // namespace

struct SpecializeEncodingsPass
    : public impl::SpecializeEncodingsPassBase<SpecializeEncodingsPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp moduleOp = getOperation();
    auto usedDialects =
        gatherUsedDialectInterfaces<AffinityAnalysisDialectInterface>(moduleOp);
    if (usedDialects.size() != 1) {
      moduleOp.emitError("expected single resolver");
      return signalPassFailure();
    }

    SymbolTable symbolTable(moduleOp);
    llvm::MapVector<StringRef, IREE::Stream::ExecutableOp> executableOps;
    for (auto executableOp : moduleOp.getOps<IREE::Stream::ExecutableOp>()) {
      executableOps[executableOp.getName()] = executableOp;
    }

    std::function<LogicalResult(AffinityAttr, Operation *,
                                SetVector<Attribute> &)>
        resolver = usedDialects[0]->makeTargetResolver(moduleOp);
    IRRewriter rewriter(&getContext());
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      SmallVector<AffinityOpInterface> candidates;
      funcOp.walk([&](AffinityOpInterface affinityOp) {
        auto affAttr = affinityOp.getAffinityAttr();
        if (!affAttr) {
          return;
        }
        candidates.push_back(affinityOp);
      });

      for (auto affinityOp : candidates) {
        auto affAttr = affinityOp.getAffinityAttr();
        // TODO: Add implementation for other ops when needed.
        LogicalResult result =
            TypeSwitch<Operation *, LogicalResult>(affinityOp)
                .Case<Stream::TensorSizeOfOp>([&](auto sizeOfOp) {
                  auto encodingType =
                      dyn_cast<RankedTensorType>(sizeOfOp.getEncoding());
                  if (!encodingType) {
                    return success();
                  }
                  auto encoding =
                      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
                          encodingType.getEncoding());
                  if (!encoding) {
                    return success();
                  }

                  SetVector<Attribute> vec;
                  // if (failed(resolver(affAttr, sizeOfOp, vec))) {
                  if (failed(resolver(affAttr, moduleOp, vec))) {
                    affinityOp.emitError("failed on getting target resolvers");
                    return failure();
                  }

                  SmallVector<Attribute> targets(vec.begin(), vec.end());
                  rewriter.modifyOpInPlace(sizeOfOp, [&] {
                    auto newEncoding = encoding.cloneWithTargets(targets);
                    sizeOfOp.setEncoding(RankedTensorType::get(
                        encodingType.getShape(), encodingType.getElementType(),
                        newEncoding));
                  });

                  return success();
                })
                .Default([](auto *op) { return success(); });

        if (failed(result)) {
          return signalPassFailure();
        }
      }
    }

    // gather per-export [execution affinity -> [resource affinities]] map
    {
      AffinityAnalysis affinityAnalysis(moduleOp);
      (void)affinityAnalysis.run();
      SmallVector<AsyncDispatchOp> candidates;
      for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
        funcOp.walk([&](AsyncDispatchOp op) { candidates.push_back(op); });
      }
      // export -> [affinity -> array per resource of affinities PVS]
      DenseMap<Stream::ExecutableExportOp,
               SetVector<std::pair<AffinityAttr, ArrayAttr>>>
          exportDispatchSites;

      llvm::MapVector<Stream::AsyncDispatchOp, SmallVector<Attribute>>
          operandsResourceAffinities;
      for (auto dispatchOp : candidates) {
        SmallVector<IREE::Stream::AffinityAttr> affinities;
        assert(affinityAnalysis.tryLookupExecutionAffinity(dispatchOp,
                                                           affinities));
        assert(affinities.size() == 1);

        SmallVector<Attribute> operandAttrs =
            getOperandsResourceAffinities(affinityAnalysis, dispatchOp);
        operandsResourceAffinities[dispatchOp] = operandAttrs;

        SymbolRefAttr entryPoint =
            *dispatchOp.getEntryPoints().getAsRange<SymbolRefAttr>().begin();
        auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
            symbolTable.lookupSymbolIn(moduleOp, entryPoint));

        exportDispatchSites[exportOp].insert(
            std::make_pair(affinities[0], ArrayAttr::get(ctx, operandAttrs)));
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Dump of exportDispatchSites\n";
        for (auto [exportOp, vec] : exportDispatchSites) {
          llvm::dbgs() << "  ExportOp: " << exportOp.getSymName() << "\n";
          for (auto [affinityAttr, arrayAttr] : vec) {
            llvm::dbgs() << "    affinity: " << affinityAttr << "\n";
            llvm::dbgs() << "    operandsResource: " << arrayAttr << "\n";
          }
        }
      });

      // Duplicate executables for each unqiue resource affinities.
      IRRewriter rewriter(ctx);
      DenseMap<std::tuple<AffinityAttr, ArrayAttr, Stream::ExecutableExportOp>,
               Stream::ExecutableOp>
          dispatchSiteToExecutable;
      for (auto [exportOp, vec] : exportDispatchSites) {
        if (vec.size() == 1) {
          continue;
        }
        int64_t dupId = -1;
        for (auto it : vec) {
          auto executableOp = exportOp->getParentOfType<Stream::ExecutableOp>();
          rewriter.setInsertionPointAfter(executableOp);
          Stream::ExecutableOp dupOp = executableOp;
          if (dupId != -1) {
            std::string symName = std::string(executableOp.getSymName());
            symName += "_" + std::to_string(dupId);
            dupOp = rewriter.cloneWithoutRegions(executableOp);
            rewriter.modifyOpInPlace(dupOp, [&] {
              dupOp.setSymName(symName);
              IRMapping mapping;
              executableOp.getRegion().cloneInto(&dupOp.getRegion(), mapping);
            });
          }
          dispatchSiteToExecutable[std::make_tuple(
              std::get<0>(it), std::get<1>(it), exportOp)] = dupOp;
          dupId++;
        }
      }

      // Update dispatch sites.
      for (auto dispatchOp : candidates) {
        SmallVector<IREE::Stream::AffinityAttr> affinities;
        assert(affinityAnalysis.tryLookupExecutionAffinity(dispatchOp,
                                                           affinities));

        SmallVector<Attribute> operandAttrs =
            operandsResourceAffinities[dispatchOp];
        SymbolRefAttr entryPoint =
            *dispatchOp.getEntryPoints().getAsRange<SymbolRefAttr>().begin();
        auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
            symbolTable.lookupSymbolIn(moduleOp, entryPoint));

        auto info = std::make_tuple(
            affinities[0], ArrayAttr::get(ctx, operandAttrs), exportOp);

        if (!dispatchSiteToExecutable.count(info)) {
          LLVM_DEBUG(llvm::dbgs() << "not found, skip\n  "
                                  << dispatchOp.getEntryPoints() << "\n");
          continue;
        }

        auto executableOp = dispatchSiteToExecutable[info];
        rewriter.modifyOpInPlace(dispatchOp, [&] {
          SmallVector<Attribute> entryPoints;
          auto newSym =
              SymbolRefAttr::get(executableOp->getAttrOfType<StringAttr>(
                                     SymbolTable::getSymbolAttrName()),
                                 entryPoint.getNestedReferences());
          entryPoints.push_back(newSym);
          dispatchOp.setEntryPointsAttr(rewriter.getArrayAttr(entryPoints));
        });
      }

      // Attach encoding targets to all the executables.
      for (auto dispatchOp : candidates) {
        SmallVector<IREE::Stream::AffinityAttr> affinities;
        assert(affinityAnalysis.tryLookupExecutionAffinity(dispatchOp,
                                                           affinities));
        SymbolRefAttr entryPoint =
            *dispatchOp.getEntryPoints().getAsRange<SymbolRefAttr>().begin();
        auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
            symbolTable.lookupSymbolIn(moduleOp, entryPoint));
        auto executableOp =
            exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
        SmallVector<Attribute> operandAttrs =
            operandsResourceAffinities[dispatchOp];
        updateExecutableOpEncodings(moduleOp, executableOp, operandAttrs,
                                    affinities[0], symbolTable, resolver);
      }
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream

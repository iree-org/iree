// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/TargetRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::stablehlo {

LogicalResult
TargetRegistry::addCustomCall(StringRef name, CustomCallHandler handler,
                              DialectRegistration registerDialects) {
  if (name.empty() || !handler) {
    return failure();
  }
  auto guardedHandler = [handler = std::move(handler)](
                            mlir::stablehlo::CustomCallOp op,
                            PatternRewriter &rewriter) -> LogicalResult {
    auto hasOrderingType = [](Type type) {
      SmallVector<Type> leaves;
      if (auto tuple = dyn_cast<TupleType>(type)) {
        tuple.getFlattenedTypes(leaves);
      } else {
        leaves.push_back(type);
      }
      return llvm::any_of(leaves, [](Type leaf) {
        return isa<mlir::stablehlo::TokenType, mlir::stablehlo::FutureType>(
            leaf);
      });
    };
    if (op.getHasSideEffect() || !op.getOutputOperandAliases().empty() ||
        llvm::any_of(op.getOperandTypes(), hasOrderingType) ||
        llvm::any_of(op.getResultTypes(), hasOrderingType)) {
      return failure();
    }
    return handler(op, rewriter);
  };
  return success(
      customCalls
          .try_emplace(name,
                       Entry<CustomCallHandler>{std::move(guardedHandler),
                                                std::move(registerDialects)})
          .second);
}

LogicalResult
TargetRegistry::addComposite(StringRef name, int32_t version,
                             CompositeHandler handler,
                             DialectRegistration registerDialects) {
  if (name.empty() || !handler) {
    return failure();
  }
  return success(
      composites[name]
          .emplace(version,
                   Entry<CompositeHandler>{std::move(handler),
                                           std::move(registerDialects)})
          .second);
}

const TargetRegistry::CustomCallHandler *
TargetRegistry::lookupCustomCall(StringRef name) const {
  auto it = customCalls.find(name);
  return it == customCalls.end() ? nullptr : &it->second.handler;
}

const TargetRegistry::CompositeHandler *
TargetRegistry::lookupComposite(StringRef name, int32_t version) const {
  auto versions = composites.find(name);
  if (versions == composites.end()) {
    return nullptr;
  }
  auto it = versions->second.find(version);
  return it == versions->second.end() ? nullptr : &it->second.handler;
}

void TargetRegistry::registerDependentDialects(
    DialectRegistry &dialects) const {
  for (const auto &customCall : customCalls) {
    if (customCall.second.registerDialects) {
      customCall.second.registerDialects(dialects);
    }
  }
  for (const auto &versions : composites) {
    for (const auto &[version, entry] : versions.second) {
      if (entry.registerDialects) {
        entry.registerDialects(dialects);
      }
    }
  }
}

namespace {
using RegistrySnapshot = std::shared_ptr<const TargetRegistry>;

struct CustomCallDispatch final
    : OpRewritePattern<mlir::stablehlo::CustomCallOp> {
  CustomCallDispatch(MLIRContext *context, RegistrySnapshot registry)
      : OpRewritePattern(context), registry(std::move(registry)) {}
  LogicalResult matchAndRewrite(mlir::stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    auto *handler = registry->lookupCustomCall(op.getCallTargetName());
    return handler ? (*handler)(op, rewriter) : failure();
  }
  RegistrySnapshot registry;
};

// Counts the leading composite operands whose decomposition argument carries
// jax.global_constant: the dimension variables and platform index JAX passes.
static unsigned countGlobalConstantOperands(mlir::stablehlo::CompositeOp op) {
  auto decomposition = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      op, op.getDecompositionAttr());
  if (!decomposition) {
    return 0;
  }
  unsigned limit =
      std::min(decomposition.getNumArguments(), op->getNumOperands());
  unsigned count = 0;
  while (count < limit &&
         decomposition.getArgAttr(count, "jax.global_constant")) {
    ++count;
  }
  return count;
}

struct CompositeDispatch final
    : OpRewritePattern<mlir::stablehlo::CompositeOp> {
  CompositeDispatch(MLIRContext *context, RegistrySnapshot registry)
      : OpRewritePattern(context), registry(std::move(registry)) {}
  LogicalResult matchAndRewrite(mlir::stablehlo::CompositeOp op,
                                PatternRewriter &rewriter) const override {
    auto *handler = registry->lookupComposite(op.getName(), op.getVersion());
    if (!handler) {
      return failure();
    }
    // Ops the handler creates carry the composite's name and numerical profile
    // so later diagnostics can name their origin.
    std::string provenance = ("composite " + op.getName()).str();
    if (auto profile = op.getCompositeAttributes().getAs<StringAttr>(
            "numerical_profile")) {
      provenance += (" (" + profile.getValue() + ")").str();
    }
    // A handler takes the composite's own operands; the original keeps the
    // global constants its decomposition needs if the handler declines.
    unsigned globalConstants = countGlobalConstantOperands(op);
    mlir::stablehlo::CompositeOp target = op;
    if (globalConstants) {
      rewriter.setInsertionPoint(op);
      target = mlir::stablehlo::CompositeOp::create(
          rewriter, op.getLoc(), op.getResultTypes(),
          op.getInputs().drop_front(globalConstants), op->getAttrs());
      rewriter.replaceAllUsesWith(op->getResults(), target->getResults());
    }
    OpBuilder::Listener *listener = rewriter.getListener();
    ProvenanceListener provenanceListener(listener,
                                          rewriter.getStringAttr(provenance));
    rewriter.setListener(&provenanceListener);
    LogicalResult result = (*handler)(target, rewriter);
    rewriter.setListener(listener);
    if (target != op) {
      if (succeeded(result)) {
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceAllUsesWith(target->getResults(), op->getResults());
        rewriter.eraseOp(target);
      }
    }
    return result;
  }

  // Names the composite in the location of every op a handler inserts.
  struct ProvenanceListener final : RewriterBase::ForwardingListener {
    ProvenanceListener(OpBuilder::Listener *listener, StringAttr provenance)
        : ForwardingListener(listener), provenance(provenance) {}
    void notifyOperationInserted(Operation *op,
                                 OpBuilder::InsertPoint previous) override {
      // A moved op keeps its location.
      if (!previous.isSet()) {
        op->setLoc(NameLoc::get(provenance, op->getLoc()));
      }
      ForwardingListener::notifyOperationInserted(op, previous);
    }
    StringAttr provenance;
  };
  RegistrySnapshot registry;
};

void populatePatterns(RewritePatternSet &patterns, RegistrySnapshot registry) {
  patterns.add<CustomCallDispatch, CompositeDispatch>(patterns.getContext(),
                                                      registry);
}

class RewriteRegisteredTargetsPass final
    : public PassWrapper<RewriteRegisteredTargetsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RewriteRegisteredTargetsPass)
  explicit RewriteRegisteredTargetsPass(const TargetRegistry &registry)
      : registry(std::make_shared<const TargetRegistry>(registry)) {}
  RewriteRegisteredTargetsPass(const RewriteRegisteredTargetsPass &) = default;

  StringRef getArgument() const override {
    return "iree-stablehlo-rewrite-registered-targets";
  }
  StringRef getDescription() const override {
    return "Attempts registered StableHLO custom-call and composite rewrites";
  }
  void getDependentDialects(DialectRegistry &dialects) const override {
    dialects.insert<mlir::stablehlo::StablehloDialect>();
    registry->registerDependentDialects(dialects);
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populatePatterns(patterns, registry);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  RegistrySnapshot registry;
};
} // namespace

void populateTargetRewritePatterns(RewritePatternSet &patterns,
                                   const TargetRegistry &registry) {
  populatePatterns(patterns, std::make_shared<const TargetRegistry>(registry));
}

std::unique_ptr<Pass>
createRewriteRegisteredTargetsPass(const TargetRegistry &registry) {
  return std::make_unique<RewriteRegisteredTargetsPass>(registry);
}

} // namespace mlir::iree_compiler::stablehlo

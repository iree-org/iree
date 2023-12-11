// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTraits.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_VERIFYINPUTPASS
#define GEN_PASS_DEF_VERIFYLOWERINGTOTENSORSPASS
#define GEN_PASS_DEF_VERIFYLOWERINGTOASYNCPASS
#define GEN_PASS_DEF_VERIFYLOWERINGTOCMDPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Base pass utility
//===----------------------------------------------------------------------===//

class Verifier {
public:
  enum class Legality {
    LEGAL,
    RECURSIVELY_LEGAL,
    ILLEGAL,
  };

  using OpVerifierFn = std::function<std::optional<Legality>(Operation *op)>;
  using TypeVerifierFn = std::function<Legality(Type type)>;

  void addIllegalDialect(StringRef dialectName) {
    dialectLegality.insert({dialectName, Legality::ILLEGAL});
  }
  template <typename DialectT>
  void addIllegalDialect() {
    addIllegalDialect(DialectT::getDialectNamespace());
  }

  template <typename OpT>
  void addLegalOp() {
    opLegality.insert({OpT::getOperationName(), Legality::LEGAL});
  }

  template <typename OpT>
  void addRecursivelyLegalOp() {
    opLegality.insert({OpT::getOperationName(), Legality::RECURSIVELY_LEGAL});
  }

  template <typename OpT>
  void addIllegalOp() {
    opLegality.insert({OpT::getOperationName(), Legality::ILLEGAL});
  }

  void addOpVerifier(std::function<std::optional<Legality>(Operation *)> fn) {
    opVerifiers.push_back(fn);
  }

  template <typename OpT>
  void addOpVerifier(std::function<std::optional<Legality>(OpT)> fn) {
    auto wrapperFn = [=](Operation *baseOp) -> std::optional<Legality> {
      if (auto op = dyn_cast<OpT>(baseOp)) {
        return fn(op);
      }
      return std::nullopt;
    };
    opVerifiers.push_back(wrapperFn);
  }

  template <typename TypeT>
  void addIllegalType() {
    typeLegality.insert({TypeID::get<TypeT>(), Legality::ILLEGAL});
  }

  template <typename TypeT>
  void addTypeVerifier(std::function<Legality(TypeT)> fn) {
    auto wrapperFn = [=](Type baseType) {
      return fn(llvm::cast<TypeT>(baseType));
    };
    if (typeVerifiers.insert({TypeID::get<TypeT>(), wrapperFn}).second ==
        false) {
      assert(false && "already registered for this type");
    }
  }

  LogicalResult run(Operation *rootOp) {
    bool foundAnyIllegal = false;
    rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto walkResult = WalkResult::advance();

      // Check for op legality - can skip the expensive work if known-illegal.
      auto legality = getOpLegality(op);
      switch (legality) {
      case Legality::LEGAL:
        // Op itself is legal but may not have valid operands/results.
        break;
      case Legality::RECURSIVELY_LEGAL:
        // If the entire op w/ nested ops is legal then skip.
        return WalkResult::skip();
      default:
      case Legality::ILLEGAL:
        // Early-exit on illegal ops without recursing.
        emitIllegalOpError(op);
        foundAnyIllegal = true;
        return WalkResult::skip();
      }

      // Check types for operands/results.
      for (auto operandType : llvm::enumerate(op->getOperandTypes())) {
        if (isTypeLegal(operandType.value()))
          continue;
        emitIllegalTypeError(op, "operand", operandType.index(),
                             operandType.value());
        foundAnyIllegal = true;
      }
      for (auto resultType : llvm::enumerate(op->getResultTypes())) {
        if (isTypeLegal(resultType.value()))
          continue;
        emitIllegalTypeError(op, "result", resultType.index(),
                             resultType.value());
        foundAnyIllegal = true;
      }

      return walkResult;
    });
    return success(!foundAnyIllegal);
  }

private:
  Legality getOpLegality(Operation *op) {
    auto opName = op->getName();

    // Check specific ops first (we may override dialect settings).
    {
      auto legalityIt = opLegality.find(opName.getStringRef());
      if (legalityIt != opLegality.end()) {
        return legalityIt->second;
      }
    }

    // Check all op verifiers (usually used for interface checks).
    for (auto &opVerifier : opVerifiers) {
      auto legalOr = opVerifier(op);
      if (legalOr.has_value()) {
        return legalOr.value();
      }
    }

    // If no op carveout is applied then check to see if the dialect is
    // allowed at all.
    {
      auto legalityIt = dialectLegality.find(opName.getDialectNamespace());
      if (legalityIt != dialectLegality.end()) {
        return legalityIt->second;
      }
    }

    // Assume legal by default.
    return Legality::LEGAL;
  }

  bool isTypeLegal(Type type) {
    // TODO(benvanik): subelements interface checks using recursive legality.

    // Defer to verifiers first.
    auto it = typeVerifiers.find(type.getTypeID());
    if (it != typeVerifiers.end()) {
      return it->second(type) != Legality::ILLEGAL;
    }

    // Check legality of the base type.
    {
      auto legalityIt = typeLegality.find(type.getTypeID());
      if (legalityIt != typeLegality.end()) {
        return legalityIt->second != Legality::ILLEGAL;
      }
    }

    // Assume legal by default.
    return true;
  }

  void emitIllegalOpError(Operation *op) {
    op->emitOpError()
        << "illegal for this phase of lowering in the stream dialect; "
           "expected to have been converted or removed";
  }

  void emitIllegalTypeError(Operation *op, StringRef location, unsigned idx,
                            Type type) {
    op->emitOpError()
        << location << " " << idx << " type " << type
        << " illegal for this phase of lowering in the stream dialect";
  }

  DenseMap<StringRef, Legality> dialectLegality;
  DenseMap<StringRef, Legality> opLegality;
  SmallVector<OpVerifierFn> opVerifiers;
  DenseMap<TypeID, Legality> typeLegality;
  DenseMap<TypeID, TypeVerifierFn> typeVerifiers;
};

static void setupDefaultOpLegality(Verifier &verifier) {
  verifier.addRecursivelyLegalOp<IREE::HAL::ExecutableOp>();
}

static void markStreamTensorOpsIllegal(Verifier &verifier) {
  verifier.addOpVerifier(
      [](Operation *op) -> std::optional<Verifier::Legality> {
        if (op->hasTrait<OpTrait::IREE::Stream::TensorPhaseOp>()) {
          return Verifier::Legality::ILLEGAL;
        }
        return std::nullopt;
      });
}

static void markStreamAsyncOpsIllegal(Verifier &verifier) {
  verifier.addOpVerifier(
      [](Operation *op) -> std::optional<Verifier::Legality> {
        if (op->hasTrait<OpTrait::IREE::Stream::AsyncPhaseOp>()) {
          return Verifier::Legality::ILLEGAL;
        }
        return std::nullopt;
      });
}

//===----------------------------------------------------------------------===//
// --iree-stream-verify-input
//===----------------------------------------------------------------------===//

struct VerifyInputPass
    : public IREE::Stream::impl::VerifyInputPassBase<VerifyInputPass> {
  void runOnOperation() override {
    Verifier verifier;
    setupDefaultOpLegality(verifier);

    // TODO(#7432): add indirect global expansion support to streams.
    verifier.addIllegalOp<IREE::Util::GlobalAddressOp>();
    verifier.addIllegalOp<IREE::Util::GlobalLoadIndirectOp>();
    verifier.addIllegalOp<IREE::Util::GlobalStoreIndirectOp>();

    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-verify-lowering-to-tensors
//===----------------------------------------------------------------------===//

static void markTensorInputsIllegal(Verifier &verifier) {
  // Tensorish dialects should all be either converted or outlined into
  // executables. Everything should be in resources now.
  verifier.addIllegalDialect("tensor");
  verifier.addIllegalDialect("linalg");

  // We don't allow the flow dialect except for inside of executables where
  // we don't yet have a full mapping to in the stream dialect (and may never).
  // Ideally we'd not be using the flow ops inside at all at this point but
  // that'd require some upstream ops (or something in codegen) for the tensor
  // load and store behaviors as well as the workgroup info.
  verifier.addIllegalDialect("flow");
  verifier.addRecursivelyLegalOp<IREE::Stream::ExecutableOp>();
}

struct VerifyLoweringToTensorsPass
    : public IREE::Stream::impl::VerifyLoweringToTensorsPassBase<
          VerifyLoweringToTensorsPass> {
  void runOnOperation() override {
    // We cannot have stream.cmd.* ops mixed with stream.tensor/async.* ops
    // as they use different memory models. We need to allow them through,
    // though, to allow for compiler re-entrancy.
    Verifier verifier;
    setupDefaultOpLegality(verifier);
    markTensorInputsIllegal(verifier);
    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-verify-lowering-to-tensors
//===----------------------------------------------------------------------===//

struct VerifyLoweringToAsyncPass
    : public IREE::Stream::impl::VerifyLoweringToAsyncPassBase<
          VerifyLoweringToAsyncPass> {
  void runOnOperation() override {
    // We cannot have stream.cmd.* ops mixed with stream.tensor/async.* ops
    // as they use different memory models. We need to allow them through,
    // though, to allow for compiler re-entrancy.
    Verifier verifier;
    setupDefaultOpLegality(verifier);
    markTensorInputsIllegal(verifier);
    markStreamTensorOpsIllegal(verifier);

    // All resources should have had their usage assigned.
    verifier.addTypeVerifier<IREE::Stream::ResourceType>([](auto type) {
      if (type.getLifetime() == IREE::Stream::Lifetime::Unknown) {
        return Verifier::Legality::ILLEGAL;
      }
      return Verifier::Legality::LEGAL;
    });

    // All streamable ops should be inside of execution regions.
    verifier.addOpVerifier<IREE::Stream::StreamableOpInterface>(
        [](auto op) -> std::optional<Verifier::Legality> {
          // Skip cmd ops that may exist.
          if (op->template hasTrait<OpTrait::IREE::Stream::CmdPhaseOp>()) {
            return Verifier::Legality::LEGAL;
          }

          // Allow metadata ops outside of execution regions.
          if (op.isMetadata())
            return Verifier::Legality::LEGAL;

          // TODO(benvanik): execution region interface to make this generic.
          if (!op->template getParentOfType<IREE::Stream::AsyncExecuteOp>()) {
            op->emitOpError()
                << ": streamable op expected to be in an execution region";
            return Verifier::Legality::ILLEGAL;
          }
          return std::nullopt;
        });

    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-stream-verify-lowering-to-cmd
//===----------------------------------------------------------------------===//

struct VerifyLoweringToCmdPass
    : public IREE::Stream::impl::VerifyLoweringToCmdPassBase<
          VerifyLoweringToCmdPass> {
  void runOnOperation() override {
    Verifier verifier;
    setupDefaultOpLegality(verifier);
    markTensorInputsIllegal(verifier);
    markStreamTensorOpsIllegal(verifier);
    markStreamAsyncOpsIllegal(verifier);

    // All resources should have had their usage assigned.
    verifier.addTypeVerifier<IREE::Stream::ResourceType>([](auto type) {
      if (type.getLifetime() == IREE::Stream::Lifetime::Unknown) {
        return Verifier::Legality::ILLEGAL;
      }
      return Verifier::Legality::LEGAL;
    });

    if (failed(verifier.run(getOperation()))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream

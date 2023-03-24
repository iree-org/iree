// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/ResourceUsage.h"

#include <utility>

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Element.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/Solver.h"
#include "iree/compiler/Dialect/Util/Analysis/DFX/State.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "iree-util-dfx"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Stream {

// TODO(benvanik): pick a policy for whether we want to favor copying external
// values into transients or try to reuse the external values. In very loopy
// programs enabling this lets us use a lot more stream-ordered allocations but
// an allocation hoisting pass would be able to do the same thing. Until we have
// that pass we can evaluate the difference manually with this flag. This could
// likely be solved by adding a NOT_LOOP_CARRIED bit to the usage and setting it
// on any value that ends up on a back edge of the CFG. We'd then favor those
// as transients instead of straight-line escaping results.
static constexpr bool kFavorTransients = false;

// Starts by assuming that the resource is never used and then removes assumed
// bits based on the usage in the program.
//
// BitIntegerState starts with all bits assumed so we invert the usage bits
// such that each bit indicates that some particular usage is _not_ performed.
// As the solver runs the assumed bits are removed each time the resource is
// used in a particular way (if the resource is read as part of a transfer
// operation then the NOT_TRANSFER_READ assumed bit will be removed). Upon
// completion we'll know for each resource what _is not_ performed and thus by
// inverting the bits we can arrive at what _is_ performed.
//
// Best state: never used at all (never read/written/etc).
// Worst state: used for all kinds of things.
template <typename ElementT>
class AbstractResourceUsage
    : public DFX::StateWrapper<DFX::BitIntegerState<uint16_t, 4095, 0>,
                               ElementT> {
 public:
  using BaseType =
      DFX::StateWrapper<DFX::BitIntegerState<uint16_t, 4095, 0>, ElementT>;

  // Inverted bits matching ResourceUsageBitfield.
  enum {
    NOT_INDIRECT = 1u << 0,
    NOT_EXTERNAL = 1u << 1,
    NOT_MUTATED = 1u << 2,  // beyond definition
    NOT_CONSTANT = 1u << 3,
    NOT_TRANSFER_READ = 1u << 4,
    NOT_TRANSFER_WRITE = 1u << 5,
    NOT_STAGING_READ = 1u << 6,
    NOT_STAGING_WRITE = 1u << 7,
    NOT_DISPATCH_READ = 1u << 8,
    NOT_DISPATCH_WRITE = 1u << 9,
    NOT_GLOBAL_READ = 1u << 10,
    NOT_GLOBAL_WRITE = 1u << 11,

    BEST_STATE = NOT_INDIRECT | NOT_EXTERNAL | NOT_MUTATED | NOT_CONSTANT |
                 NOT_TRANSFER_READ | NOT_TRANSFER_WRITE | NOT_STAGING_READ |
                 NOT_STAGING_WRITE | NOT_DISPATCH_READ | NOT_DISPATCH_WRITE |
                 NOT_GLOBAL_READ | NOT_GLOBAL_WRITE,
  };
  static_assert(BEST_STATE == BaseType::getBestState(),
                "unexpected BEST_STATE value");

  static bool isValidStateBits(uint16_t bits) {
    // bool isIndirect = (bits & NOT_INDIRECT) != NOT_INDIRECT;
    // bool isExternal = (bits & NOT_EXTERNAL) != NOT_EXTERNAL;
    bool isMutated = (bits & NOT_MUTATED) != NOT_MUTATED;
    bool isConstant = (bits & NOT_CONSTANT) != NOT_CONSTANT;
    // bool isTransferRead = (bits & NOT_TRANSFER_READ) != NOT_TRANSFER_READ;
    // bool isTransferWrite = (bits & NOT_TRANSFER_WRITE) != NOT_TRANSFER_WRITE;
    bool isStagingRead = (bits & NOT_STAGING_READ) != NOT_STAGING_READ;
    bool isStagingWrite = (bits & NOT_STAGING_WRITE) != NOT_STAGING_WRITE;
    bool isDispatchRead = (bits & NOT_DISPATCH_READ) != NOT_DISPATCH_READ;
    bool isDispatchWrite = (bits & NOT_DISPATCH_WRITE) != NOT_DISPATCH_WRITE;
    // bool isGlobalRead = (bits & NOT_GLOBAL_READ) != NOT_GLOBAL_READ;
    // bool isGlobalWrite = (bits & NOT_GLOBAL_WRITE) != NOT_GLOBAL_WRITE;

    // Cannot be both staging and dispatch.
    if ((isStagingRead || isStagingWrite) &&
        (isDispatchRead || isDispatchWrite)) {
      return false;
    }

    // Cannot be both constant and mutated.
    // TODO(benvanik): maybe allow this for initializers that perform dispatches
    // to initialize the resources. This introduces copies of those that are
    // annoying to elide later on.
    if (isConstant && isMutated) {
      return false;
    }

    return true;
  }

  bool isValidState() const override {
    return this->getAssumed() != BaseType::getWorstState() &&
           isValidStateBits(this->getAssumed());
  }

  ResourceUsageBitfield convertBitsToResourceUsage(uint16_t bits) const {
    return static_cast<ResourceUsageBitfield>(~bits & BEST_STATE);
  }

  ResourceUsageBitfield getKnownUsage() const {
    return convertBitsToResourceUsage(this->getKnown());
  }

  ResourceUsageBitfield getAssumedUsage() const {
    return convertBitsToResourceUsage(this->getAssumed());
  }

  const std::string getAsStr(AsmState &asmState) const override {
    std::string str;
    if (!isValidState()) return "*";
    auto append = [&](const char *part) {
      if (!str.empty()) str += '|';
      str += part;
    };
    if (!this->isAssumed(NOT_INDIRECT)) append("indirect");
    append(this->isAssumed(NOT_EXTERNAL) ? "internal" : "external");
    append(this->isAssumed(NOT_MUTATED) ? "immutable" : "mutable");
    if (!this->isAssumed(NOT_CONSTANT)) append("constant");
    if (!this->isAssumed(NOT_TRANSFER_READ)) append("transfer_read");
    if (!this->isAssumed(NOT_TRANSFER_WRITE)) append("transfer_write");
    if (!this->isAssumed(NOT_STAGING_READ)) append("staging_read");
    if (!this->isAssumed(NOT_STAGING_WRITE)) append("staging_write");
    if (!this->isAssumed(NOT_DISPATCH_READ)) append("dispatch_read");
    if (!this->isAssumed(NOT_DISPATCH_WRITE)) append("dispatch_write");
    if (!this->isAssumed(NOT_GLOBAL_READ)) append("global_read");
    if (!this->isAssumed(NOT_GLOBAL_WRITE)) append("global_write");
    return str.empty() ? "*" : str;
  }

 protected:
  explicit AbstractResourceUsage(const Position &pos) : BaseType(pos) {}

  // Add known bits based on the static type information available.
  // Doing this sets the worst case bits that analysis cannot remove.
  void initializeFromType(IREE::Stream::ResourceType type) {
    BaseType::intersectAssumedBits(BEST_STATE);
    switch (type.getLifetime()) {
      case Lifetime::Unknown:
        break;
      case Lifetime::External:
        BaseType::intersectAssumedBits(BEST_STATE & ~NOT_EXTERNAL);
        BaseType::addKnownBits(NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Staging:
        BaseType::intersectAssumedBits(
            BEST_STATE & (~NOT_STAGING_READ | ~NOT_STAGING_WRITE |
                          ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_DISPATCH_READ |
                               NOT_DISPATCH_WRITE | NOT_GLOBAL_READ |
                               NOT_GLOBAL_WRITE);
        break;
      case Lifetime::Transient:
        BaseType::intersectAssumedBits(
            BEST_STATE & (~NOT_DISPATCH_READ | ~NOT_DISPATCH_WRITE |
                          ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Variable:
        BaseType::intersectAssumedBits(
            BEST_STATE & (~NOT_GLOBAL_READ | ~NOT_GLOBAL_WRITE |
                          ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_EXTERNAL | NOT_CONSTANT | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
      case Lifetime::Constant:
        BaseType::intersectAssumedBits(
            BEST_STATE &
            (~NOT_CONSTANT | ~NOT_TRANSFER_READ | ~NOT_TRANSFER_WRITE));
        BaseType::addKnownBits(NOT_MUTATED | NOT_EXTERNAL | NOT_STAGING_READ |
                               NOT_STAGING_WRITE);
        break;
    }
  }
};

// Starts with the best assumed state of the value never being used for anything
// and then works towards a worst state of it being used for everything.
class ValueResourceUsage : public AbstractResourceUsage<DFX::ValueElement> {
 public:
  using BaseType = AbstractResourceUsage<DFX::ValueElement>;

  static ValueResourceUsage &createForPosition(const Position &pos,
                                               DFX::Solver &solver) {
    return *(new (solver.getAllocator()) ValueResourceUsage(pos));
  }

  const std::string getName() const override { return "ValueResourceUsage"; }
  const void *getID() const override { return &ID; }
  static bool classof(const DFX::AbstractElement *element) {
    return (element->getID() == &ID);
  }

  static const char ID;

 private:
  explicit ValueResourceUsage(const Position &pos) : BaseType(pos) {}

  // Starts analysis of the |value| with known bits based on its resource type.
  void initializeValue(Value value, DFX::Solver &solver) override {
    auto resourceType = value.getType().cast<IREE::Stream::ResourceType>();
    initializeFromType(resourceType);
  }

  // Updates the usage based on the op defining the value.
  // This may be dynamic as the result value may be tied to an operand that
  // itself is under analysis.
  void updateFromDefiningOp(Value value, OpResult result, DFX::Solver &solver) {
    // Some tied uses route through ops that change types - ignore those.
    if (!result.getType().isa<IREE::Stream::ResourceType>()) return;

    TypeSwitch<Operation *, void>(result.getOwner())
        .Case([&](mlir::arith::SelectOp op) {
          auto trueUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTrueValue()),
              DFX::Resolution::REQUIRED);
          auto falseUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getFalseValue()),
              DFX::Resolution::REQUIRED);
          getState() ^= trueUsage.getState();
          getState() ^= falseUsage.getState();
        })
        .Case([&](IREE::Util::OptimizationBarrierOp op) {
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getOperand(0)),
              DFX::Resolution::REQUIRED);
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Util::GlobalLoadOpInterface op) {
          removeAssumedBits(NOT_GLOBAL_READ);
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.getGlobalName(), op);
          auto globalType = globalInfo->op.getGlobalType()
                                .template cast<IREE::Stream::ResourceType>();
          switch (globalType.getLifetime()) {
            case IREE::Stream::Lifetime::Constant:
              removeAssumedBits(NOT_CONSTANT);
              break;
            case IREE::Stream::Lifetime::Variable:
            default:
              break;
          }
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getLoadedGlobalValue()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Util::GlobalLoadIndirectOpInterface op) {
          removeAssumedBits(NOT_INDIRECT | NOT_GLOBAL_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getLoadedGlobalValue()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::ResourceStoreOp op) {
          removeAssumedBits(NOT_STAGING_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTarget()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::TensorImportOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_EXTERNAL);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncConstantOp op) {
          removeAssumedBits(NOT_CONSTANT | NOT_TRANSFER_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncSplatOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncCloneOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getSource()),
              DFX::Resolution::OPTIONAL);
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncSliceOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getSource()),
              DFX::Resolution::OPTIONAL);
          getState() ^= sourceUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncFillOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTarget()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncUpdateOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTarget()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncCopyOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTarget()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncCollectiveOp op) {
          // We treat collectives as transfer + dispatch as any particular
          // implementation may use either (or both).
          // TODO(#11249): handle source == target aliasing.
          removeAssumedBits(NOT_TRANSFER_WRITE | NOT_DISPATCH_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTarget()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncTransferOp op) {
          removeAssumedBits(NOT_TRANSFER_WRITE);
          auto sourceUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getSource()),
              DFX::Resolution::OPTIONAL);
          bool isSourceStaging = !(sourceUsage.isAssumed(NOT_STAGING_READ) &&
                                   sourceUsage.isAssumed(NOT_STAGING_WRITE));
          bool isTargetStaging =
              !(isAssumed(NOT_STAGING_READ) && isAssumed(NOT_STAGING_WRITE));
          if (isSourceStaging != isTargetStaging) {
            // Can't transition staging across transfers.
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping transfer source: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          // TODO(benvanik): remove kFavorTransients.
          bool isSourceExternal = !sourceUsage.isAssumed(NOT_EXTERNAL);
          bool isTargetInternal = isAssumed(NOT_EXTERNAL);
          if (kFavorTransients && isSourceExternal && isTargetInternal) {
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping forward prop of "
                              "external into internal: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          auto newState = getState();
          newState ^= sourceUsage.getState();
          if (!newState.isValidState()) {
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping update from "
                              "producer as it would create an invalid state: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          getState() = newState;
        })
        .Case([&](IREE::Stream::AsyncStoreOp op) {
          removeAssumedBits(NOT_STAGING_WRITE);
          auto targetUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getTarget()),
              DFX::Resolution::REQUIRED);
          getState() ^= targetUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncDispatchOp op) {
          removeAssumedBits(NOT_DISPATCH_WRITE);
          auto tiedOperand = op.getTiedResultOperand(result);
          if (tiedOperand) {
            auto tiedUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(tiedOperand),
                DFX::Resolution::REQUIRED);
            getState() ^= tiedUsage.getState();
          } else {
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(result), DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncCallOp op) {
          // We treat calls as transfer + dispatch as any particular callee may
          // use either (or both).
          removeAssumedBits(NOT_TRANSFER_WRITE | NOT_DISPATCH_WRITE);
          auto tiedOperand = op.getTiedResultOperand(result);
          if (tiedOperand) {
            auto tiedUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(tiedOperand),
                DFX::Resolution::REQUIRED);
            getState() ^= tiedUsage.getState();
          } else {
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(result), DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Default([&](Operation *op) {});
  }

  // Updates the usage based on the particular usage as |operand|.
  // This walks through tied uses as well.
  void updateFromUse(Value value, OpOperand &operand, DFX::Solver &solver) {
    // Some tied uses route through ops that change types - ignore those.
    if (!operand.get().getType().isa<IREE::Stream::ResourceType>()) return;

    auto *userOp = operand.getOwner();
    unsigned operandIdx = operand.getOperandNumber();
    TypeSwitch<Operation *, void>(userOp)
        .Case([&](mlir::arith::SelectOp op) {
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](mlir::BranchOpInterface op) {
          auto operandUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op->getOperand(operandIdx)),
              DFX::Resolution::REQUIRED);
          getState() ^= operandUsage.getState();
          solver.getExplorer().walkOutgoingBranchOperandArguments(
              op, operandIdx, [&](Block *targetBlock, BlockArgument arg) {
                auto argUsage = solver.getElementFor<ValueResourceUsage>(
                    *this, Position::forValue(arg), DFX::Resolution::OPTIONAL);
                getState() ^= argUsage;
                return WalkResult::advance();
              });
        })
        .Case([&](mlir::func::ReturnOp op) {
          auto operandUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getOperand(operandIdx)),
              DFX::Resolution::REQUIRED);
          getState() ^= operandUsage.getState();
          solver.getExplorer().walkIncomingCalls(
              op->getParentOfType<mlir::CallableOpInterface>(),
              [&](mlir::CallOpInterface callOp) {
                auto argUsage = solver.getElementFor<ValueResourceUsage>(
                    *this, Position::forValue(callOp->getResult(operandIdx)),
                    DFX::Resolution::OPTIONAL);
                getState() ^= argUsage;
                return WalkResult::advance();
              });
        })
        .Case([&](IREE::Util::OptimizationBarrierOp op) {
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult(0)),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Util::GlobalStoreOpInterface op) {
          removeAssumedBits(NOT_GLOBAL_WRITE);
          auto *globalInfo =
              solver.getExplorer().queryGlobalInfoFrom(op.getGlobalName(), op);
          auto globalType = globalInfo->op.getGlobalType()
                                .template cast<IREE::Stream::ResourceType>();
          switch (globalType.getLifetime()) {
            case IREE::Stream::Lifetime::Constant:
              removeAssumedBits(NOT_CONSTANT);
              break;
            case IREE::Stream::Lifetime::Variable:
            default:
              break;
          }
        })
        .Case([&](IREE::Util::GlobalStoreIndirectOpInterface op) {
          removeAssumedBits(NOT_INDIRECT | NOT_GLOBAL_WRITE);
        })
        .Case([&](IREE::Stream::TensorExportOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_EXTERNAL);
        })
        .Case([&](IREE::Stream::AsyncCloneOp op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::OPTIONAL);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncSliceOp op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::OPTIONAL);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncFillOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncUpdateOp op) {
          if (value == op.getUpdate()) {
            removeAssumedBits(NOT_TRANSFER_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.getResult()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncCopyOp op) {
          if (value == op.getSource()) {
            removeAssumedBits(NOT_TRANSFER_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.getResult()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncCollectiveOp op) {
          // We treat collectives as transfer + dispatch as any particular
          // implementation may use either (or both).
          // TODO(#11249): handle source == target aliasing.
          if (value == op.getSource()) {
            removeAssumedBits(NOT_TRANSFER_READ | NOT_DISPATCH_READ);
          } else {
            removeAssumedBits(NOT_MUTATED | NOT_TRANSFER_WRITE |
                              NOT_DISPATCH_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(op.getResult()),
                DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncTransferOp op) {
          removeAssumedBits(NOT_TRANSFER_READ);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::OPTIONAL);
          bool isSourceStaging =
              !(isAssumed(NOT_STAGING_READ) && isAssumed(NOT_STAGING_WRITE));
          bool isTargetStaging = !(resultUsage.isAssumed(NOT_STAGING_READ) &&
                                   resultUsage.isAssumed(NOT_STAGING_WRITE));
          if (isSourceStaging != isTargetStaging) {
            // Can't transition staging across transfers.
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping transfer target: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          // TODO(benvanik): remove kFavorTransients.
          bool isSourceInternal = isAssumed(NOT_EXTERNAL);
          bool isTargetExternal = !resultUsage.isAssumed(NOT_EXTERNAL);
          if (kFavorTransients && isSourceInternal && isTargetExternal) {
            LLVM_DEBUG({
              llvm::dbgs()
                  << "[ValueResourceUsage] skipping back prop of external into "
                     "internal due to kFavorTransients: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          auto newState = getState();
          newState ^= resultUsage.getState();
          if (!newState.isValidState()) {
            LLVM_DEBUG({
              llvm::dbgs() << "[ValueResourceUsage] skipping update from use "
                              "as it would create an invalid state: ";
              op.print(llvm::dbgs(), solver.getAsmState());
              llvm::dbgs() << "\n";
            });
            return;
          }
          getState() = newState;
        })
        .Case([&](IREE::Stream::AsyncLoadOp op) {
          removeAssumedBits(NOT_STAGING_READ);
        })
        .Case([&](IREE::Stream::AsyncStoreOp op) {
          removeAssumedBits(NOT_MUTATED | NOT_STAGING_WRITE);
          auto resultUsage = solver.getElementFor<ValueResourceUsage>(
              *this, Position::forValue(op.getResult()),
              DFX::Resolution::REQUIRED);
          getState() ^= resultUsage.getState();
        })
        .Case([&](IREE::Stream::AsyncDispatchOp op) {
          removeAssumedBits(NOT_DISPATCH_READ);
          for (auto result : op.getOperandTiedResults(operandIdx)) {
            removeAssumedBits(NOT_MUTATED | NOT_DISPATCH_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(result), DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Case([&](IREE::Stream::AsyncCallOp op) {
          // We treat calls as transfer + dispatch as any particular callee may
          // use either (or both).
          removeAssumedBits(NOT_TRANSFER_READ | NOT_DISPATCH_READ);
          for (auto result : op.getOperandTiedResults(operandIdx)) {
            removeAssumedBits(NOT_MUTATED | NOT_DISPATCH_WRITE);
            auto resultUsage = solver.getElementFor<ValueResourceUsage>(
                *this, Position::forValue(result), DFX::Resolution::REQUIRED);
            getState() ^= resultUsage.getState();
          }
        })
        .Default([&](Operation *op) {});
  }

  // Updates the usage state of |value| by walking all defining ops (up through
  // function arguments, branch arguments, and tied results) and all transitive
  // uses (down through function calls, branches, and tied operands).
  ChangeStatus updateValue(Value value, DFX::Solver &solver) override {
    auto assumedBits = getAssumed();

    auto traversalResult = TraversalResult::COMPLETE;

    // Join with defining ops - of which there may be multiple if we come from
    // a branch/region argument or call result.
    traversalResult |=
        solver.getExplorer().walkDefiningOps(value, [&](OpResult result) {
          updateFromDefiningOp(value, result, solver);
          return WalkResult::advance();
        });

    // Join with using ops.
    traversalResult |=
        solver.getExplorer().walkTransitiveUses(value, [&](OpOperand &operand) {
          updateFromUse(value, operand, solver);
          return WalkResult::advance();
        });

    if (traversalResult == TraversalResult::INCOMPLETE) {
      removeAssumedBits(NOT_EXTERNAL);
    }

    return assumedBits == getAssumed() ? ChangeStatus::UNCHANGED
                                       : ChangeStatus::CHANGED;
  }

  friend class DFX::Solver;
};
const char ValueResourceUsage::ID = 0;

ResourceUsageAnalysis::ResourceUsageAnalysis(Operation *rootOp)
    : explorer(rootOp, TraversalAction::SHALLOW), solver(explorer, allocator) {
  explorer.setOpAction<IREE::Util::InitializerOp>(TraversalAction::RECURSE);
  explorer.setOpAction<mlir::func::FuncOp>(TraversalAction::RECURSE);
  explorer.setDialectAction<IREE::Stream::StreamDialect>(
      TraversalAction::RECURSE);
  // Ignore the contents of executables (linalg goo, etc).
  explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
  explorer.initialize();
}

ResourceUsageAnalysis::~ResourceUsageAnalysis() = default;

std::optional<ResourceUsageBitfield>
ResourceUsageAnalysis::tryLookupResourceUsage(Value value) {
  auto resourceUsage =
      solver.lookupElementFor<ValueResourceUsage>(Position::forValue(value));
  if (!resourceUsage) return std::nullopt;
  return resourceUsage->getAssumedUsage();
}

LogicalResult ResourceUsageAnalysis::run() {
  // TODO(benvanik): initialize globals and track usage through them.
  // Today we pin globals to <constant> or <variable> but it'd be nice to
  // set that based on actual usage here.
  //
  // Initialize globals that we need to resolve.
  // explorer.forEachGlobal([&](const auto *globalInfo) {
  //   auto globalType = globalInfo->op.type();
  //   if (globalType.template isa<IREE::Stream::ResourceType>()) {
  //     solver.getOrCreateElementFor<GlobalResourceUsage>(
  //         Position::forOperation(globalInfo->op));
  //   }
  // });

  // Initialize all SSA values we can do just with trivial search.
  explorer.walkValues([&](Value value) {
    if (value.getType().isa<IREE::Stream::ResourceType>()) {
      solver.getOrCreateElementFor<ValueResourceUsage>(
          Position::forValue(value));
    }
    return WalkResult::advance();
  });

  return solver.run();
}

}  // namespace Stream
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

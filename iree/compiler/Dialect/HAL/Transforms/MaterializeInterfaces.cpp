// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-hal-materialize-interfaces"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// hal.executable.variant creation
//===----------------------------------------------------------------------===//

// Creates zero or more hal.executable.variant ops for each target backend.
// The source op will contain the flow.executable contents and any attributes
// the backend wants to carry along during transformation.
static LogicalResult declareVariantOps(TargetOptions targetOptions,
                                       IREE::Flow::ExecutableOp sourceOp,
                                       IREE::HAL::ExecutableOp executableOp) {
  // The user has specified what targets they want as a set of patterns. This
  // matches against those patterns so vulkan-* may match vulkan-v1.1 and
  // vulkan-v1.2.
  auto targetBackends = matchTargetBackends(targetOptions.targets);
  if (targetBackends.empty()) {
    auto diagnostic = sourceOp.emitError();
    diagnostic
        << "no target backends available for executable translation; ensure "
        << "they are linked in and the target options are properly "
        << "specified. requested = [ ";
    for (const auto &target : targetOptions.targets) {
      diagnostic << "'" << target << "' ";
    }
    diagnostic << "], available = [ ";
    for (const auto &target : getRegisteredTargetBackends()) {
      diagnostic << "'" << target << "' ";
    }
    diagnostic << "]";
    return diagnostic;
  }

  // Materialize all of the hal.executable.variant ops for all backends we are
  // targeting. Note that each backend may create zero or more target ops.
  for (auto &targetBackend : targetBackends) {
    targetBackend->declareVariantOps(sourceOp, executableOp);
  }

  // Ensure that at least one target op got created. If it didn't that means
  // the executable cannot be translated and it's better to fail now.
  if (executableOp.getBlock()
          .getOps<IREE::HAL::ExecutableVariantOp>()
          .empty()) {
    auto diagnostic = sourceOp.emitError();
    diagnostic
        << "no target backend was able to handle this executable; tried = [ ";
    for (const auto &target : targetOptions.targets) {
      diagnostic << "'" << target << "' ";
    }
    diagnostic << "]";
    return diagnostic;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Derived dispatch usage information
//===----------------------------------------------------------------------===//

class OperandResultRef {
 public:
  static OperandResultRef makeOperand(int index) {
    return OperandResultRef{0, index};
  }
  static OperandResultRef makeResult(int index) {
    return OperandResultRef{1, index};
  }

  OperandResultRef() : type(0), value(0) {}
  OperandResultRef(int type, int value) : type(type), value(value) {}

  bool isOperand() const { return type == 0; }
  bool isResult() const { return type == 1; }
  int index() const { return value; }

 private:
  int type : 1;
  int value : 31;
};

// HACK: only needed because we don't have the right IR ops to represent this.
// I hate this.
static OperandResultRef mapRegionOperandToDispatchValue(
    FuncOp entryFuncOp, int regionOperandIndex) {
  auto types = entryFuncOp.getType().getInputs();
  int operandIndex = 0;
  int resultIndex = 0;
  for (int i = 0; i < regionOperandIndex; ++i) {
    auto operandType = types[i];
    if (operandType.isIntOrIndexOrFloat()) {
      ++operandIndex;
    } else if (auto tensorType =
                   operandType.dyn_cast<IREE::Flow::DispatchTensorType>()) {
      switch (tensorType.getAccess()) {
        case IREE::Flow::TensorAccess::ReadOnly:
          ++operandIndex;
          break;
        case IREE::Flow::TensorAccess::ReadWrite:
          ++operandIndex;
          ++resultIndex;
          break;
        case IREE::Flow::TensorAccess::WriteOnly:
          ++resultIndex;
          break;
      }
    }
  }
  auto operandType = types[regionOperandIndex];
  if (operandType.isIntOrIndexOrFloat()) {
    return OperandResultRef::makeOperand(operandIndex);
  } else if (auto tensorType =
                 operandType.dyn_cast<IREE::Flow::DispatchTensorType>()) {
    switch (tensorType.getAccess()) {
      case IREE::Flow::TensorAccess::ReadOnly:
        return OperandResultRef::makeOperand(operandIndex);
      case IREE::Flow::TensorAccess::ReadWrite:
        return OperandResultRef::makeOperand(operandIndex);
      case IREE::Flow::TensorAccess::WriteOnly:
        return OperandResultRef::makeResult(resultIndex);
    }
  }
  llvm_unreachable("invalid region operand/result mapping");
}

// Information about dispatch region operand usage derived from flow.dispatches.
struct DerivedOperandInfo {
  enum Usage : unsigned {
    // Unused or not yet scanned.
    UNKNOWN = 0,
    // I/O from a constant pool storage buffer.
    CONSTANT_BUFFER = 1u << 0,
    // I/O to a transient storage buffer.
    TRANSIENT_BUFFER = 1u << 1,
    // I/O to an external/unknowable buffer (invocation capture/escape).
    EXTERNAL_BUFFER = 1u << 2,
    // Push constant (index) non-buffer type.
    PUSH_CONSTANT = 1u << 3,
  };

  // Operand index in the flattened dispatch function.
  int operandIndex = 0;

  // Bitfield of usage throughout the program.
  // Entry points dispatched from many places may have differing usage.
  unsigned usage = Usage::UNKNOWN;

  // Storage buffer and ranges pulled from it if this has CONSTANT_BUFFER usage
  // and all uses are of the same storage buffer. Unset if multiple storage
  // buffers are used.
  SymbolRefAttr constantBuffer;
  SmallVector<Attribute> constantRanges;

  void dump() {
    if (usage & DerivedOperandInfo::Usage::PUSH_CONSTANT) {
      llvm::dbgs() << "  PUSH_CONSTANT\n";
    }
    if (usage & DerivedOperandInfo::Usage::CONSTANT_BUFFER) {
      if (constantBuffer) {
        llvm::dbgs() << "  CONSTANT_BUFFER " << constantBuffer << "\n";
        for (auto &range : constantRanges) {
          llvm::dbgs() << "   RANGE " << range << "\n";
        }
      } else {
        llvm::dbgs() << "  CONSTANT_BUFFER (NON-UNIFORM STORAGE)\n";
      }
    }
    if (usage & DerivedOperandInfo::Usage::TRANSIENT_BUFFER) {
      llvm::dbgs() << "  TRANSIENT_BUFFER\n";
    }
    if (usage & DerivedOperandInfo::Usage::EXTERNAL_BUFFER) {
      llvm::dbgs() << "  EXTERNAL_BUFFER\n";
    }
  }
};

// Derives information about each dispatch region operand based on all of the
// |dispatchOps| which dispatch it. Returns a vector mapped 1:1 with the
// |entryFuncOp| operand list of flow.dispatch.tensor types.
static SmallVector<DerivedOperandInfo> deriveOperandInfo(
    FuncOp entryFuncOp, ArrayRef<IREE::Flow::DispatchOp> dispatchOps) {
  // Mapping of flattened dispatch region operands to binding information.
  SmallVector<DerivedOperandInfo> operandInfos;
  operandInfos.resize(entryFuncOp.getNumArguments());

  SmallVector<OperandResultRef> regionToDispatchMap;
  regionToDispatchMap.resize(entryFuncOp.getNumArguments());
  for (int i = 0; i < entryFuncOp.getNumArguments(); ++i) {
    regionToDispatchMap[i] = mapRegionOperandToDispatchValue(entryFuncOp, i);
  }

  for (auto dispatchOp : dispatchOps) {
    for (unsigned operandIndex = 0;
         operandIndex < entryFuncOp.getNumArguments(); ++operandIndex) {
      auto &operandInfo = operandInfos[operandIndex];
      operandInfo.operandIndex = operandIndex;

      auto ref = regionToDispatchMap[operandIndex];
      if (!ref.isOperand()) {
        // Results always go to external buffers.
        // TODO(benvanik): mark as transient if local to the stream.
        operandInfo.usage |= DerivedOperandInfo::Usage::EXTERNAL_BUFFER;
        continue;
      }

      auto operand = dispatchOp.operands()[ref.index()];
      auto operandType = operand.getType();
      if (operandType.isIntOrIndexOrFloat()) {
        operandInfo.usage = DerivedOperandInfo::Usage::PUSH_CONSTANT;
      } else if (operandType.isa<TensorType>()) {
        if (auto subspanOp = dyn_cast_or_null<IREE::HAL::ConstantSubspanOp>(
                operand.getDefiningOp())) {
          // Operand comes from a constant storage buffer. There may be
          // multiple of these buffers so we have to check the reference and
          // bucket on that.
          if ((operandInfo.usage &
               DerivedOperandInfo::Usage::CONSTANT_BUFFER) == 0) {
            // First use as a constant.
            assert(!operandInfo.constantBuffer);
            operandInfo.usage |= DerivedOperandInfo::Usage::CONSTANT_BUFFER;
            operandInfo.constantBuffer = subspanOp.runtime_buffer();
            operandInfo.constantRanges.push_back(subspanOp.runtime_range());
          } else if (operandInfo.constantBuffer == subspanOp.runtime_buffer()) {
            // Same storage buffer but (possibly) unique range.
            operandInfo.constantRanges.push_back(subspanOp.runtime_range());
          } else {
            // Has already been used as a constant but with a different
            // buffer. We flag these to indicate that though it's sourced
            // from a constant buffer we can't uniformly bind it. We could
            // do some more sophisticated scanning here and see if all
            // constants from this single dispatch come from the same buffer
            // and at least treat those as uniform but for now we bail.
            operandInfo.constantBuffer = nullptr;
            operandInfo.constantRanges.clear();
          }
        } else {
          // Today we aren't tracking transient buffers yet and all are
          // assumed to be external.
          operandInfo.usage |= DerivedOperandInfo::Usage::EXTERNAL_BUFFER;
        }
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Derived binding info for @" << entryFuncOp.getName()
                 << ":\n";
    for (auto info : operandInfos) {
      llvm::dbgs() << " operand " << info.operandIndex << ": "
                   << entryFuncOp.getType().getInput(info.operandIndex) << "\n";
      info.dump();
    }
  });

  return operandInfos;
}

//===----------------------------------------------------------------------===//
// Computed binding and operand information
//===----------------------------------------------------------------------===//

// Constant binding info for a storage buffer used by a single dispatch.
struct ConstantBindingUsage {
  // Storage buffer symbol.
  SymbolRefAttr constantBuffer;
  // Minimum offset of all usage across all operands.
  int64_t minimumOffset = INT64_MAX;
  // Maximum offset of all usage across all operands.
  int64_t maximumOffset = INT64_MIN;
};

// Defines an offset applied to a binding.
struct BindingOffset {
  static constexpr int64_t INVALID_OFFSET = INT64_MAX;

  // Static relative byte offset always applied to the binding.
  int64_t staticOffset = INVALID_OFFSET;

  // TODO(benvanik): dynamic offset/offset selection based on usage.
  // For example we may have a [usage index -> static offset] map that we can
  // emit as a lookup table or some affine expression for computing the offset.
};

// Defines a dispatch region operand binding mapping.
// Multiple operands may share the same binding with differing offsets.
struct RegionOperand {
  enum class Type {
    UNSPECIFIED = 0,
    PUSH_CONSTANT = 1,
    BUFFER = 2,
  };

  // Specifies what type of binding this is.
  Type type = Type::UNSPECIFIED;

  // Push constant ordinal if Type::PUSH_CONSTANT;
  int pushConstantOrdinal = 0;

  // Binding on the interface if Type::BUFFER.
  IREE::HAL::InterfaceBindingOp bindingOp;
  // Offset applied to the operand base binding.
  BindingOffset bindingOffset;
};

// Per-usage binding information.
// A single binding may cover multiple operands.
struct DispatchBinding {
  enum class Type {
    UNUSED = 0,
    CONSTANT_STORAGE = 1,
    TRANSIENT_BUFFER = 2,
    EXTERNAL_BUFFER = 3,
    PUSH_CONSTANT = 4,
  };

  // Specifies what type of binding this is (indicating which usage info field
  // is populated).
  Type type = Type::UNUSED;

  // Binding on the interface if a buffer/storage type.
  IREE::HAL::InterfaceBindingOp bindingOp;

  // Push constant ordinal if Type::PUSH_CONSTANT.
  // TODO(#5322): replace with InterfaceConstantOp.
  int pushConstantOrdinal = 0;

  // Type::CONSTANT_STORAGE information defining the constant storage buffer
  // that should be bound and what range is used. Each usage site may bind a
  // different range of the same constant storage buffer.
  ConstantBindingUsage constant;

  // TODO(benvanik): drop when using a flattened operand list like dispatch
  // entry points - then we can reuse the region operand index directly.
  llvm::Optional<OperandResultRef> sourceIndex;

  void dump() {
    if (type == Type::UNUSED) {
      llvm::dbgs() << "  UNUSED\n";
      return;
    } else if (type == Type::PUSH_CONSTANT) {
      llvm::dbgs() << "PUSH_CONSTANT " << pushConstantOrdinal << " = ";
      if (sourceIndex.getValue().isOperand()) {
        llvm::dbgs() << "operand[" << sourceIndex.getValue().index() << "]";
      } else {
        llvm::dbgs() << "result[" << sourceIndex.getValue().index() << "]";
      }
      llvm::dbgs() << "\n";
      return;
    }

    llvm::dbgs() << "@" << bindingOp.sym_name() << ": ";
    if (type == Type::CONSTANT_STORAGE) {
      llvm::dbgs() << "CONSTANT_STORAGE " << constant.constantBuffer << " +"
                   << constant.minimumOffset;
    } else {
      if (type == Type::TRANSIENT_BUFFER) {
        llvm::dbgs() << "TRANSIENT_BUFFER";
      } else {
        llvm::dbgs() << "EXTERNAL_BUFFER";
      }
      llvm::dbgs() << " = ";
      if (sourceIndex.getValue().isOperand()) {
        llvm::dbgs() << "operand[" << sourceIndex.getValue().index() << "]";
      } else {
        llvm::dbgs() << "result[" << sourceIndex.getValue().index() << "]";
      }
    }
    llvm::dbgs() << "\n";
  }
};

// Utility for building hal.interface ops by accumulating bindings.
// Handles uniquing the descriptor ordinals and bucketing into sets by usage.
class InterfaceBuilder {
 public:
  explicit InterfaceBuilder(FuncOp entryFuncOp,
                            IREE::HAL::InterfaceOp interfaceOp)
      : entryFuncOp(entryFuncOp),
        interfaceOp(interfaceOp),
        builder(OpBuilder::atBlockBegin(&interfaceOp.body().front())) {
    regionOperands.resize(entryFuncOp.getNumArguments());
  }

  // Defines a new binding on the interface with the given properties.
  // The symbol name will be automatically deduped.
  IREE::HAL::InterfaceBindingOp defineBinding(
      DispatchBinding::Type usageType, IREE::HAL::DescriptorType descriptorType,
      IREE::HAL::MemoryAccessBitfield memoryAccess) {
    // TODO(benvanik): pick set based on usage type; constants/transients in
    // one, external in another, etc.
    int setOrdinal = 0;

    std::string bindingName = "s" + std::to_string(setOrdinal) + "b" +
                              std::to_string(nextBindingOrdinal);
    if (allEnumBitsSet(memoryAccess,
                       IREE::HAL::MemoryAccessBitfield::Read |
                           IREE::HAL::MemoryAccessBitfield::Write)) {
      bindingName += "_rw";
    } else if (allEnumBitsSet(memoryAccess,
                              IREE::HAL::MemoryAccessBitfield::Read)) {
      bindingName += "_ro";
    } else if (allEnumBitsSet(memoryAccess,
                              IREE::HAL::MemoryAccessBitfield::Discard |
                                  IREE::HAL::MemoryAccessBitfield::Write)) {
      bindingName += "_xw";
    } else if (allEnumBitsSet(memoryAccess,
                              IREE::HAL::MemoryAccessBitfield::Write)) {
      bindingName += "_wo";
    }
    switch (usageType) {
      case DispatchBinding::Type::CONSTANT_STORAGE:
        bindingName += "_constant";
        break;
      case DispatchBinding::Type::TRANSIENT_BUFFER:
        bindingName += "_transient";
        break;
      case DispatchBinding::Type::EXTERNAL_BUFFER:
        bindingName += "_external";
        break;
      default:
        llvm_unreachable("unhandled usage type");
        break;
    }

    auto bindingOp = builder.create<IREE::HAL::InterfaceBindingOp>(
        interfaceOp.getLoc(), bindingName, /*set=*/APInt(64, setOrdinal),
        /*binding=*/APInt(64, nextBindingOrdinal++), descriptorType,
        memoryAccess);
    return bindingOp;
  }

  // TODO(#5322): make push constants symbolic so that this returns a symref.
  int definePushConstant() {
    int ordinal = nextPushConstantOrdinal++;
    interfaceOp.push_constantsAttr(
        builder.getIndexAttr(nextPushConstantOrdinal));
    return ordinal;
  }

  // Maps an operand of the entry point to a binding at a particular offset.
  // The mapping can be applied to the function with applyRegionMappings.
  void mapRegionOperand(int operandIndex, RegionOperand regionOperand) {
    regionOperands[operandIndex] = std::move(regionOperand);
  }

  // Maps a binding to a fixed value or operand/result of |dispatchOp|.
  // These parameters will be used to map the operands to HAL bindings during
  // conversion.
  void mapUsageBinding(IREE::Flow::DispatchOp dispatchOp,
                       DispatchBinding dispatchBinding) {
    auto &usage = usageBindings[dispatchOp.getOperation()];
    usage.push_back(dispatchBinding);
  }

  // Returns a list of dispatch region operand indices that are not yet mapped.
  // TODO(benvanik): remove this and just map them all by construction. This is
  // a fallback for when the other cases are not handled.
  SmallVector<int> getUnmappedRegionOperandIndices() {
    SmallVector<int> indices;
    for (int i = 0; i < regionOperands.size(); ++i) {
      if (regionOperands[i].type == RegionOperand::Type::UNSPECIFIED) {
        indices.push_back(i);
      }
    }
    return indices;
  }

  // Applies all mappings made via mapRegionOperand to the entry function and
  // replaces all operands with hal.interface.* ops.
  // Returns a free-floating function.
  FuncOp buildRegionFuncOp();

  // Applies all mappings made via mapUsageOperands/mapUsageResults to all
  // dispatch sites using interface.
  void applyUsageMappings();

  void dump();

 private:
  FuncOp entryFuncOp;
  IREE::HAL::InterfaceOp interfaceOp;
  OpBuilder builder;

  int nextBindingOrdinal = 0;
  int nextPushConstantOrdinal = 0;

  // One per operand on the dispatch region entry point defining operand types.
  SmallVector<RegionOperand> regionOperands;

  // Operand and result mappings for each dispatch site.
  // TODO(benvanik): just have operands that match 1:1 with regionOperands.
  DenseMap<Operation *, SmallVector<DispatchBinding>> usageBindings;
};

void InterfaceBuilder::dump() {
  interfaceOp.dump();

  llvm::dbgs() << "applyRegionMappings: @" << entryFuncOp.sym_name() << "\n";
  for (auto regionOperand : llvm::enumerate(regionOperands)) {
    llvm::dbgs() << "  operand[" << regionOperand.index() << "]: ";
    auto &value = regionOperand.value();
    switch (value.type) {
      case RegionOperand::Type::PUSH_CONSTANT:
        llvm::dbgs() << "PUSH_CONSTANT +" << value.pushConstantOrdinal;
        break;
      case RegionOperand::Type::BUFFER:
        llvm::dbgs() << "BUFFER @" << value.bindingOp.sym_name();
        if (value.bindingOffset.staticOffset != BindingOffset::INVALID_OFFSET) {
          llvm::dbgs() << " +" << value.bindingOffset.staticOffset;
        }
        break;
      default:
        llvm_unreachable("unhandled operand type");
        break;
    }
    llvm::dbgs() << "\n";
  }

  llvm::dbgs() << "applyUsageMappings: @" << entryFuncOp.sym_name() << "\n";
  for (auto it : usageBindings) {
    auto *operation = it.getFirst();
    llvm::dbgs() << "  ";
    operation->dump();
    auto &usage = it.getSecond();
    for (int i = 0; i < usage.size(); ++i) {
      llvm::dbgs() << "    ";
      usage[i].dump();
    }
  }
}

FuncOp InterfaceBuilder::buildRegionFuncOp() {
  // Clone so that we can do a bunch of unsafe in-place updates.
  auto clonedFuncOp = entryFuncOp.clone();

  // Strip all arguments as functions take all I/O through the interface API.
  clonedFuncOp.setType(FunctionType::get(clonedFuncOp.getContext(), {}, {}));

  auto *entryBlock = &clonedFuncOp.front();
  OpBuilder entryBuilder = OpBuilder::atBlockBegin(entryBlock);

  // +0 offset is used a lot.
  auto zeroOffset = entryBuilder.createOrFold<mlir::ConstantIndexOp>(
      clonedFuncOp.getLoc(), 0);

  for (auto regionOperand : llvm::enumerate(regionOperands)) {
    auto blockArg = entryBlock->getArgument(regionOperand.index());
    auto &value = regionOperand.value();
    switch (value.type) {
      case RegionOperand::Type::PUSH_CONSTANT: {
        auto loadOp = entryBuilder.create<IREE::HAL::InterfaceLoadConstantOp>(
            clonedFuncOp.getLoc(), blockArg.getType(),
            APInt(64, value.pushConstantOrdinal));
        blockArg.replaceAllUsesWith(loadOp);
        break;
      }
      case RegionOperand::Type::BUFFER: {
        Value offset = zeroOffset;
        if (value.bindingOffset.staticOffset != BindingOffset::INVALID_OFFSET &&
            value.bindingOffset.staticOffset != 0) {
          offset = entryBuilder.createOrFold<mlir::ConstantIndexOp>(
              clonedFuncOp.getLoc(), value.bindingOffset.staticOffset);
        }
        auto bindingSymRefAttr = entryBuilder.getSymbolRefAttr(
            interfaceOp.sym_name(),
            {entryBuilder.getSymbolRefAttr(value.bindingOp)});
        auto subspanOp =
            entryBuilder.create<IREE::HAL::InterfaceBindingSubspanOp>(
                clonedFuncOp.getLoc(), blockArg.getType(), bindingSymRefAttr,
                /*byte_offset=*/offset,
                /*byte_length=*/Value{});
        blockArg.replaceAllUsesWith(subspanOp);
        break;
      }
      default:
        llvm_unreachable("unhandled operand type");
        break;
    }
  }

  // Remove all of the arguments now that we've turned them into symbol
  // accesses and replaced their uses.
  while (entryBlock->getNumArguments() > 0) {
    entryBlock->eraseArgument(0);
  }

  return clonedFuncOp;
}

// HACK: this will be replaced with a direct IR update when we have a new
// intermediate HAL dispatch op. For now we need to spooky-action-at-a-distance
// the flow.dispatch conversion in ConvertStreamOps.
void InterfaceBuilder::applyUsageMappings() {
  Builder builder(entryFuncOp.getContext());
  for (auto it : usageBindings) {
    auto *operation = it.getFirst();
    SmallVector<Attribute> attrs;
    for (auto &usage : it.getSecond()) {
      switch (usage.type) {
        case DispatchBinding::Type::PUSH_CONSTANT:
          attrs.push_back(IREE::HAL::ExPushConstantAttr::get(
              builder.getIndexAttr(usage.pushConstantOrdinal),
              builder.getIndexAttr(usage.sourceIndex.getValue().index())));
          break;
        case DispatchBinding::Type::CONSTANT_STORAGE:
          attrs.push_back(IREE::HAL::ExConstantStorageAttr::get(
              builder.getStringAttr(usage.bindingOp.getName()),
              builder.getStringAttr(
                  usage.constant.constantBuffer.getLeafReference()),
              builder.getIndexAttr(usage.constant.minimumOffset),
              builder.getIndexAttr(usage.constant.maximumOffset -
                                   usage.constant.minimumOffset + 1)));
          break;
        case DispatchBinding::Type::TRANSIENT_BUFFER:
        case DispatchBinding::Type::EXTERNAL_BUFFER:
          if (usage.sourceIndex.getValue().isOperand()) {
            attrs.push_back(IREE::HAL::ExOperandBufferAttr::get(
                builder.getStringAttr(usage.bindingOp.getName()),
                builder.getIndexAttr(usage.sourceIndex.getValue().index())));
          } else {
            attrs.push_back(IREE::HAL::ExResultBufferAttr::get(
                builder.getStringAttr(usage.bindingOp.getName()),
                builder.getIndexAttr(usage.sourceIndex.getValue().index())));
          }
          break;
        default:
          llvm_unreachable("unhandled binding type");
          break;
      }
    }
    operation->setAttr("hal.bindings", builder.getArrayAttr(attrs));
  }
}

// Extracts a set of constant buffer bindings from derived operand usage.
// Zero or more bindings will be defined on |interfaceBuilder| and
// usage-specific mappings will be stashed for further processing.
static void defineConstantBindings(FuncOp entryFuncOp,
                                   ArrayRef<IREE::Flow::DispatchOp> dispatchOps,
                                   ArrayRef<DerivedOperandInfo> operandInfos,
                                   InterfaceBuilder &interfaceBuilder) {
  // Bucket all constant operand info by the storage buffer holding the
  // constant data contents at runtime.
  DenseMap<Attribute, SmallVector<DerivedOperandInfo>> constantInfos;
  for (auto &operandInfo : operandInfos) {
    if ((operandInfo.usage & DerivedOperandInfo::Usage::CONSTANT_BUFFER) ==
        DerivedOperandInfo::Usage::CONSTANT_BUFFER) {
      // Purely used as a constant. Operands used as both constant and dynamic
      // values aren't covered by this.
      if (!operandInfo.constantBuffer) continue;
      auto &storageInfo = constantInfos[operandInfo.constantBuffer];
      storageInfo.push_back(operandInfo);
    }
  }
  LLVM_DEBUG({
    for (auto usageInfos : constantInfos) {
      auto &usageInfo = usageInfos.getSecond();
      for (size_t i = 1; i < usageInfo.size(); ++i) {
        assert(usageInfo[i].constantRanges.size() ==
                   usageInfo[0].constantRanges.size() &&
               "expect uniform usage of operand as a constant across all "
               "dispatches");
      }
    }
  });

  // Find the min/max use range of each constant storage per usage.
  // This is the range that will be bound during the dispatch.
  DenseMap<Attribute, SmallVector<ConstantBindingUsage>> constantUsage;
  for (auto &constantInfo : constantInfos) {
    auto &storageUsages = constantUsage[constantInfo.getFirst()];
    for (auto &operandInfo : constantInfo.getSecond()) {
      storageUsages.resize(operandInfo.constantRanges.size());
      for (auto attr : llvm::enumerate(operandInfo.constantRanges)) {
        auto byteRangeAttr = attr.value().cast<ByteRangeAttr>();
        auto &usage = storageUsages[attr.index()];
        usage.constantBuffer = operandInfo.constantBuffer;
        usage.minimumOffset = std::min(byteRangeAttr.offset().getSExtValue(),
                                       usage.minimumOffset);
        usage.maximumOffset = std::max(
            (byteRangeAttr.offset() + byteRangeAttr.length()).getSExtValue() -
                1,
            usage.maximumOffset);
      }
    }
  }

  // When multiple operands are from the same constant buffer we compute a
  // relative offset between them. This only works if all usage of a storage
  // buffer is consistent throughout all usage.
  //
  // Example:
  //   operand[0]: @storage0: offset 100
  //   operand[1]: @storage0: offset 200
  //   operand[2]: @storage0: offset 300
  // ->
  //   operand[0]: @storage0: offset +0
  //   operand[1]: @storage0: offset +100
  //   operand[2]: @storage0: offset +200
  //
  // The relative value is based off of the minimum offset used in each
  // invocation and the constant buffer will be bound with that range.
  //  INT64_MIN: not yet defined
  //  INT64_MAX: non-uniform access (can't use relative strides).
  DenseMap<Attribute, SmallVector<int64_t>> operandOffsets;
  for (auto &constantInfo : constantInfos) {
    auto &storageUsages = constantUsage[constantInfo.getFirst()];
    auto &storageOffsets = operandOffsets[constantInfo.getFirst()];
    storageOffsets.resize(entryFuncOp.getNumArguments(), INT64_MIN);
    for (auto &operandInfo : constantInfo.getSecond()) {
      auto &operandOffset = storageOffsets[operandInfo.operandIndex];
      for (auto attr : llvm::enumerate(operandInfo.constantRanges)) {
        auto byteRangeAttr = attr.value().cast<ByteRangeAttr>();
        auto &usage = storageUsages[attr.index()];
        int64_t offset =
            byteRangeAttr.offset().getSExtValue() - usage.minimumOffset;
        switch (operandOffset) {
          case INT64_MIN:
            // First use; take offset directly as a starting point.
            operandOffset = offset;
            break;
          case INT64_MAX:
            // Already invalidated due to non-uniform strides.
            break;
          default:
            if (operandOffset != offset) {
              // This usage has a different stride than another making this a
              // non-uniform usage.
              // TODO(benvanik): handle these by inserting dispatch ID-based
              // selections of the unique value.
              operandOffset = INT64_MAX;
            }
            break;
        }
      }
    }

    // Fixup any that were not defined.
    for (int64_t &offset : storageOffsets) {
      if (offset == INT64_MIN) {
        offset = BindingOffset::INVALID_OFFSET;
      }
    }
  }

  LLVM_DEBUG({
    for (auto &constantInfo : constantInfos) {
      auto &storageUsages = constantUsage[constantInfo.getFirst()];
      auto &storageOffsets = operandOffsets[constantInfo.getFirst()];
      llvm::dbgs() << constantInfo.getFirst() << ":\n";
      for (size_t i = 0; i < storageUsages.size(); ++i) {
        auto &usage = storageUsages[i];
        llvm::dbgs() << "  usage[" << i << "]:\n";
        llvm::dbgs() << "    min = " << usage.minimumOffset << "\n";
        llvm::dbgs() << "    max = " << usage.maximumOffset << "\n";
      }
      for (size_t i = 0; i < entryFuncOp.getNumArguments(); ++i) {
        switch (storageOffsets[i]) {
          case INT64_MIN:
            continue;  // unused
          case INT64_MAX:
            llvm::dbgs() << "  operand[" << i << "]: NON-UNIFORM STRIDES\n";
            break;
          default:
            llvm::dbgs() << "  operand[" << i << "]: " << storageOffsets[i]
                         << "\n";
            break;
        }
      }
    }
  });

  // Construct the binding usage information for these constant bindings at each
  // usage site and setup a hal.interface.binding per constant storage.
  //
  // Any binding not covered here - such as those with non-uniform strides -
  // will be handled by the default case for transient/external buffers.
  for (auto &constantInfo : constantInfos) {
    auto &storageUsages = constantUsage[constantInfo.getFirst()];
    auto &storageOffsets = operandOffsets[constantInfo.getFirst()];

    // Skip if there are no uniform strides. It'll be treated as a normal
    // buffer.
    bool hasAnyUniformStrides = false;
    for (size_t i = 0; i < entryFuncOp.getNumArguments(); ++i) {
      if (storageOffsets[i] != BindingOffset::INVALID_OFFSET) {
        hasAnyUniformStrides = true;
        break;
      }
    }
    if (!hasAnyUniformStrides) continue;

    // Create the read-only binding for the constant storage.
    // TODO(benvanik): StorageBufferDynamic when using descriptor sets so that
    // we can change the binding base offset per usage without updating the
    // descriptor set. The total number of dynamic bindings per set is often
    // much lower than any other so we'd want to also be bucketing by set.
    auto bindingOp =
        interfaceBuilder.defineBinding(DispatchBinding::Type::CONSTANT_STORAGE,
                                       IREE::HAL::DescriptorType::StorageBuffer,
                                       IREE::HAL::MemoryAccessBitfield::Read);

    // Map usage of each operand in the region to the binding at a particular
    // offset.
    for (auto &operandInfo : constantInfo.getSecond()) {
      int64_t storageOffset = storageOffsets[operandInfo.operandIndex];
      if (storageOffset == BindingOffset::INVALID_OFFSET) continue;
      RegionOperand regionOperand;
      regionOperand.type = RegionOperand::Type::BUFFER;
      regionOperand.bindingOp = bindingOp;
      regionOperand.bindingOffset.staticOffset = storageOffset;
      interfaceBuilder.mapRegionOperand(operandInfo.operandIndex,
                                        std::move(regionOperand));
    }

    // Get the indices of the dispatch site operands that are covered by this
    // binding.
    // TODO(benvanik): remove this once dispatch and region operands match. We
    // need to get an intermediate HAL op for that or change flow.dispatch.
    SmallVector<int> usageOperandIndices;
    for (auto &operandInfo : constantInfo.getSecond()) {
      if (storageOffsets[operandInfo.operandIndex] !=
          BindingOffset::INVALID_OFFSET) {
        usageOperandIndices.push_back(operandInfo.operandIndex);
      }
    }

    // Record for each usage the new binding mapping. Each site may have its own
    // unique offsets.
    for (size_t i = 0; i < storageUsages.size(); ++i) {
      auto &usage = storageUsages[i];
      DispatchBinding dispatchBinding;
      dispatchBinding.type = DispatchBinding::Type::CONSTANT_STORAGE;
      dispatchBinding.bindingOp = bindingOp;
      dispatchBinding.constant = usage;
      interfaceBuilder.mapUsageBinding(dispatchOps[i],
                                       std::move(dispatchBinding));
    }
  }
}

//===----------------------------------------------------------------------===//
// hal.interface creation
//===----------------------------------------------------------------------===//

// Populates |interfaceBuilder| with bindings required by the dispatch derived
// from usage information from all dispatch sites.
static void populateInterfaceBindings(IREE::Flow::ExecutableOp executableOp,
                                      IREE::Flow::DispatchEntryOp entryOp,
                                      FuncOp entryFuncOp,
                                      SymbolTable &symbolTable,
                                      InterfaceBuilder &interfaceBuilder) {
  // Find all dispatches to the entry point within the program.
  // We could do this much faster - right now this is
  // O(entry_points * dispatches) and it would be better to invert this to
  // first find all dispatches and bucket by entry point.
  auto uses = symbolTable.getSymbolUses(entryOp, executableOp->getParentOp());
  if (!uses.hasValue()) return;
  SmallVector<IREE::Flow::DispatchOp> dispatchOps;
  for (auto &use : uses.getValue()) {
    if (auto dispatchOp = dyn_cast<IREE::Flow::DispatchOp>(use.getUser())) {
      dispatchOps.push_back(dispatchOp);
    }
  }

  // Derive information about each operand of entryFuncOp.
  auto operandInfos = deriveOperandInfo(entryFuncOp, dispatchOps);

  // Extract information about the constant buffers used for each dispatch.
  // Constant usage that is non-uniform across all dispatches to the entry point
  // may not be included in the set.
  defineConstantBindings(entryFuncOp, dispatchOps, operandInfos,
                         interfaceBuilder);

  // TODO(benvanik): defineTransientBindings.
  // TODO(benvanik): defineExternalBindings.

  // Setup push constants and map usage to the interface.
  // We may need to add some synthetic ones or change the push constants to pull
  // from a uniform buffer instead.
  for (auto &operandInfo : operandInfos) {
    // TODO(#5322): support synthesized constants and uniform buffer
    // spilling. For now we just reserve dense ordinals based on the order they
    // appear in the dispatch arguments. This won't support reordering or
    // combining until push constants use symbols.
    int primitiveIndex = 0;
    if (operandInfo.usage & DerivedOperandInfo::PUSH_CONSTANT) {
      RegionOperand regionOperand;
      auto pushConstantOrdinal = interfaceBuilder.definePushConstant();
      regionOperand.type = RegionOperand::Type::PUSH_CONSTANT;
      regionOperand.pushConstantOrdinal = pushConstantOrdinal;
      interfaceBuilder.mapRegionOperand(operandInfo.operandIndex,
                                        std::move(regionOperand));

      // HACK: this is because we don't yet have the right ops in place; we have
      // a pretty big abstraction gap between Flow <-> HAL and need some
      // intermediate ops that let us track this all in IR without performing
      // remapping.
      DispatchBinding dispatchBinding;
      dispatchBinding.type = DispatchBinding::Type::PUSH_CONSTANT;
      dispatchBinding.pushConstantOrdinal = pushConstantOrdinal;
      dispatchBinding.sourceIndex = mapRegionOperandToDispatchValue(
          entryFuncOp, operandInfo.operandIndex);

      // Record for each usage the new binding mapping. Each site may have its
      // own unique offsets.
      for (size_t i = 0; i < dispatchOps.size(); ++i) {
        interfaceBuilder.mapUsageBinding(dispatchOps[i], dispatchBinding);
      }

      ++primitiveIndex;
    }
  }

  // Handle any remaining operands as generic buffers. This may also catch
  // leftovers from constant/transient bindings that were non-uniform.
  for (int operandIndex : interfaceBuilder.getUnmappedRegionOperandIndices()) {
    auto operandType = entryFuncOp.getArgumentTypes()[operandIndex];
    auto tensorType = operandType.dyn_cast<IREE::Flow::DispatchTensorType>();
    if (!tensorType) continue;

    // Define binding for the buffer.
    IREE::HAL::MemoryAccessBitfield memoryAccess =
        IREE::HAL::MemoryAccessBitfield::None;
    switch (tensorType.getAccess()) {
      case IREE::Flow::TensorAccess::ReadOnly:
        memoryAccess = IREE::HAL::MemoryAccessBitfield::Read;
        break;
      case IREE::Flow::TensorAccess::ReadWrite:
        memoryAccess = IREE::HAL::MemoryAccessBitfield::Read |
                       IREE::HAL::MemoryAccessBitfield::Write;
        break;
      case IREE::Flow::TensorAccess::WriteOnly:
        memoryAccess = IREE::HAL::MemoryAccessBitfield::DiscardWrite;
        break;
    }
    auto bindingOp = interfaceBuilder.defineBinding(
        DispatchBinding::Type::EXTERNAL_BUFFER,
        IREE::HAL::DescriptorType::StorageBuffer, memoryAccess);

    // Map region operand to the binding.
    RegionOperand regionOperand;
    regionOperand.type = RegionOperand::Type::BUFFER;
    regionOperand.bindingOp = bindingOp;
    regionOperand.bindingOffset.staticOffset = 0;
    interfaceBuilder.mapRegionOperand(operandIndex, std::move(regionOperand));

    // HACK: this is because we don't yet have the right ops in place; we have
    // a pretty big abstraction gap between Flow <-> HAL and need some
    // intermediate ops that let us track this all in IR without performing
    // remapping.
    DispatchBinding dispatchBinding;
    dispatchBinding.type = DispatchBinding::Type::EXTERNAL_BUFFER;
    dispatchBinding.bindingOp = bindingOp;
    dispatchBinding.sourceIndex =
        mapRegionOperandToDispatchValue(entryFuncOp, operandIndex);

    // Record for each usage the new binding mapping. Each site may have its own
    // unique offsets.
    for (size_t i = 0; i < dispatchOps.size(); ++i) {
      interfaceBuilder.mapUsageBinding(dispatchOps[i], dispatchBinding);
    }
  }
}

//===----------------------------------------------------------------------===//
// Primary pass logic
//===----------------------------------------------------------------------===//

// Verifies that all types used with the given entry point are supportable.
static LogicalResult verifyEntryPointTypes(FuncOp entryFuncOp) {
  for (auto inputType : llvm::enumerate(entryFuncOp.getType().getInputs())) {
    if (inputType.value().isa<IREE::Flow::DispatchTensorType>()) {
      // We could verify the element type of the tensor here if we wanted but in
      // theory the creation of the dispatch region did that already.
    } else if (inputType.value().isa<IndexType>()) {
      // Index types are converted to platform bit-width later on.
      // TODO(benvanik): pick something here that the target devices support.
    } else if (auto integerType = inputType.value().dyn_cast<IntegerType>()) {
      if (integerType.getIntOrFloatBitWidth() != 32) {
        return entryFuncOp.emitError()
               << "unsupported argument " << inputType.index() << " bit depth "
               << integerType.getIntOrFloatBitWidth() << " (" << integerType
               << "); only 32-bit values are supported right now";
      }
    } else {
      return entryFuncOp.emitError()
             << "unsupported interface function argument " << inputType.index()
             << " type " << inputType.value()
             << "; requires tensors or simple primitive values (i32, etc)";
    }
  }
  return success();
}

// Adds the entry point ops with assigned ordinals for each entry function.
// The entry points will all use the provided |interfaceOp|.
static LogicalResult declareEntryPointOps(
    IREE::Flow::ExecutableOp sourceExecutableOp,
    IREE::HAL::ExecutableOp targetExecutableOp, SymbolTable &symbolTable) {
  auto variantOps =
      targetExecutableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>();
  OpBuilder executableBuilder(&targetExecutableOp.getBlock().front());

  // For each Flow entry point, create a HAL entry point and dispatch thunk.
  int nextOrdinal = 0;
  for (auto dispatchEntryOp :
       sourceExecutableOp.body().getOps<IREE::Flow::DispatchEntryOp>()) {
    int ordinal = nextOrdinal++;
    auto sourceFuncOp =
        sourceExecutableOp.getInnerModule().lookupSymbol<FuncOp>(
            dispatchEntryOp.function_ref());

    // Verify the source function types are usable.
    if (failed(verifyEntryPointTypes(sourceFuncOp))) {
      return failure();
    }

    // Create an interface builder we will use to populate the bindings.
    auto interfaceOp = executableBuilder.create<IREE::HAL::InterfaceOp>(
        sourceFuncOp.getLoc(), "io");
    InterfaceBuilder interfaceBuilder(sourceFuncOp, interfaceOp);

    // Find all uses of the entry point in the program and derive usage
    // information such as what kind of tensors are passed in.
    // Since a given entry point may be dispatched from multiple places this
    // allows us to generate an interface compatible with all of them.
    populateInterfaceBindings(sourceExecutableOp, dispatchEntryOp, sourceFuncOp,
                              symbolTable, interfaceBuilder);

    LLVM_DEBUG(interfaceBuilder.dump());

    // HACK: apply the recorded per-use mappings back to the flow.dispatch ops.
    interfaceBuilder.applyUsageMappings();

    auto baseFuncOp = interfaceBuilder.buildRegionFuncOp();
    for (auto variantOp : variantOps) {
      // Declare the entry point on the target.
      OpBuilder targetBuilder(&variantOp.getBlock().front());
      targetBuilder.create<IREE::HAL::ExecutableEntryPointOp>(
          dispatchEntryOp.getLoc(),
          targetBuilder.getStringAttr(dispatchEntryOp.function_ref()),
          targetBuilder.getIndexAttr(ordinal),
          targetBuilder.getSymbolRefAttr(interfaceOp), ArrayAttr{},
          IntegerAttr{});

      // Clone the updated interface-based function into the target.
      auto targetFuncOp = baseFuncOp.clone();
      variantOp.getInnerModule().push_back(targetFuncOp);

      // Copy interface bindings into the target module so symbol references
      // work.
      auto inlinedInterfaceOp = interfaceOp.clone();
      inlinedInterfaceOp.setPrivate();
      variantOp.getInnerModule().push_back(inlinedInterfaceOp);
    }

    baseFuncOp.erase();
  }
  return success();
}

namespace {
template <typename SrcOp, typename DstOp>
class ConverterDispatchWorkgroupInfoPattern final
    : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DstOp>(op, op.getResult().getType(),
                                       op.dimensionAttr());
    return success();
  }
};

}  // namespace

class MaterializeInterfacesPass
    : public PassWrapper<MaterializeInterfacesPass, OperationPass<ModuleOp>> {
 public:
  explicit MaterializeInterfacesPass(TargetOptions targetOptions)
      : targetOptions_(targetOptions) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackends = matchTargetBackends(targetOptions_.targets);
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

  StringRef getArgument() const override {
    return "iree-hal-materialize-interfaces";
  }

  StringRef getDescription() const override {
    return "Materializes hal.executable ops from flow.executable ops";
  }

  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    // Processes all executables within the input module and produce the
    // output HAL ops. We should ensure all deduping is performed prior to
    // this when it's easier to diff IR and where we still have the flow
    // context.
    auto sourceOps =
        llvm::to_vector<32>(getOperation().getOps<IREE::Flow::ExecutableOp>());
    for (auto sourceOp : sourceOps) {
      // Only manipulate tiled executables.
      // TODO(benvanik): remove this check once linalg-on-tensors is default.
      auto entryOps = sourceOp.getOps<IREE::Flow::DispatchEntryOp>();
      if (entryOps.empty()) continue;
      auto anyEntryOp = *entryOps.begin();
      if (!anyEntryOp.workgroup_rank().hasValue()) {
        continue;
      }

      // Create the op that will contain the translated executable.
      OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());
      builder.setInsertionPointAfter(sourceOp);
      auto executableOp = builder.create<IREE::HAL::ExecutableOp>(
          sourceOp.getLoc(), sourceOp.getName());
      executableOp.setVisibility(sourceOp.getVisibility());

      // Embed the hal.executable.variant ops for each source.
      if (failed(declareVariantOps(targetOptions_, sourceOp, executableOp))) {
        return signalPassFailure();
      }

      // Define interfaces for each entry point.
      if (failed(declareEntryPointOps(sourceOp, executableOp, symbolTable))) {
        return signalPassFailure();
      }

      // Convert interface-related flow.dispatch.* ops to their hal.*
      // versions.
      OwningRewritePatternList patterns(&getContext());
      patterns.insert<ConverterDispatchWorkgroupInfoPattern<
                          IREE::Flow::DispatchWorkgroupIDOp,
                          IREE::HAL::InterfaceWorkgroupIDOp>,
                      ConverterDispatchWorkgroupInfoPattern<
                          IREE::Flow::DispatchWorkgroupCountOp,
                          IREE::HAL::InterfaceWorkgroupCountOp>,
                      ConverterDispatchWorkgroupInfoPattern<
                          IREE::Flow::DispatchWorkgroupSizeOp,
                          IREE::HAL::InterfaceWorkgroupSizeOp>>(
          executableOp.getContext());
      if (failed(applyPatternsAndFoldGreedily(executableOp,
                                              std::move(patterns)))) {
        return signalPassFailure();
      }

      sourceOp.erase();
    }
  }

 private:
  TargetOptions targetOptions_;
};

std::unique_ptr<OperationPass<ModuleOp>> createMaterializeInterfacesPass(
    TargetOptions executableOptions) {
  return std::make_unique<MaterializeInterfacesPass>(executableOptions);
}

static PassRegistration<MaterializeInterfacesPass> pass([] {
  auto options = getTargetOptionsFromFlags();
  return std::make_unique<MaterializeInterfacesPass>(options);
});

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

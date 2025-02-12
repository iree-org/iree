// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/StringUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler::IREE::HAL {

//===----------------------------------------------------------------------===//
// Alignment
//===----------------------------------------------------------------------===//

llvm::MaybeAlign commonAlignment(llvm::MaybeAlign lhs, llvm::MaybeAlign rhs) {
  if (!lhs.has_value() || !rhs.has_value())
    return std::nullopt;
  return llvm::MaybeAlign(
      llvm::MinAlign(lhs.value().value(), rhs.value().value()));
}

// TODO(benvanik): share with align op folder and analysis.
// May need an interface for querying the alignment from ops that can carry it.
std::optional<uint64_t> lookupOffsetOrAlignment(Value value) {
  APInt constantValue;
  if (matchPattern(value, m_ConstantInt(&constantValue))) {
    // Value is constant and we can just treat that as if it were an alignment.
    return constantValue.getZExtValue();
  }

  auto op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto alignmentAttr = op->getAttrOfType<IntegerAttr>("stream.alignment")) {
    // The op has an alignment tagged on it we can use directly.
    return alignmentAttr.getValue().getZExtValue();
  }

  // TODO(benvanik): walk other pass-through. These are the most common in our
  // programs today.
  if (auto loadOp = dyn_cast<IREE::HAL::InterfaceConstantLoadOp>(op)) {
    // Push constants have an optional value alignment.
    auto alignment = loadOp.getAlignment();
    if (alignment.has_value()) {
      return alignment.value().getZExtValue();
    }
  } else if (auto castOp = dyn_cast<arith::IndexCastUIOp>(op)) {
    return lookupOffsetOrAlignment(castOp.getOperand());
  } else if (auto assumeOp = dyn_cast<IREE::Util::AssumeIntOp>(op)) {
    return assumeOp.getUnionedUnsignedDivisor(
        cast<OpResult>(value).getResultNumber());
  }

  // TODO(benvanik): more searching using util.align and other ops.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Object types
//===----------------------------------------------------------------------===//

Value BufferType::inferSizeFromValue(Location loc, Value value,
                                     OpBuilder &builder) const {
  return builder.createOrFold<BufferLengthOp>(loc, builder.getIndexType(),
                                              value);
}

Value BufferViewType::inferSizeFromValue(Location loc, Value value,
                                         OpBuilder &builder) const {
  return builder.createOrFold<BufferLengthOp>(
      loc, builder.getIndexType(),
      builder.createOrFold<BufferViewBufferOp>(
          loc, builder.getType<IREE::HAL::BufferType>(), value));
}

// static
Value DeviceType::resolveAny(Location loc, OpBuilder &builder) {
  Value deviceIndex = builder.create<arith::ConstantIndexOp>(loc, 0);
  return builder.create<IREE::HAL::DevicesGetOp>(
      loc, builder.getType<IREE::HAL::DeviceType>(), deviceIndex);
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/HAL/IR/HALOpInterfaces.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/HALTypeInterfaces.cpp.inc"

void HALDialect::registerTypes() {
  addTypes<AllocatorType, BufferType, BufferViewType, ChannelType,
           CommandBufferType, DeviceType, EventType, ExecutableType, FenceType,
           FileType, SemaphoreType>();
}

Type HALDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeKind;
  if (parser.parseKeyword(&typeKind))
    return {};
  auto type = llvm::StringSwitch<Type>(typeKind)
                  .Case("allocator", AllocatorType::get(getContext()))
                  .Case("buffer", BufferType::get(getContext()))
                  .Case("buffer_view", BufferViewType::get(getContext()))
                  .Case("channel", ChannelType::get(getContext()))
                  .Case("command_buffer", CommandBufferType::get(getContext()))
                  .Case("device", DeviceType::get(getContext()))
                  .Case("event", EventType::get(getContext()))
                  .Case("executable", ExecutableType::get(getContext()))
                  .Case("fence", FenceType::get(getContext()))
                  .Case("file", FileType::get(getContext()))
                  .Case("semaphore", SemaphoreType::get(getContext()))
                  .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown HAL type: " << typeKind;
  }
  return type;
}

void HALDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (llvm::isa<AllocatorType>(type)) {
    p << "allocator";
  } else if (llvm::isa<BufferType>(type)) {
    p << "buffer";
  } else if (llvm::isa<BufferViewType>(type)) {
    p << "buffer_view";
  } else if (llvm::isa<ChannelType>(type)) {
    p << "channel";
  } else if (llvm::isa<CommandBufferType>(type)) {
    p << "command_buffer";
  } else if (llvm::isa<DeviceType>(type)) {
    p << "device";
  } else if (llvm::isa<EventType>(type)) {
    p << "event";
  } else if (llvm::isa<ExecutableType>(type)) {
    p << "executable";
  } else if (llvm::isa<FenceType>(type)) {
    p << "fence";
  } else if (llvm::isa<FileType>(type)) {
    p << "file";
  } else if (llvm::isa<SemaphoreType>(type)) {
    p << "semaphore";
  } else {
    assert(false && "unknown HAL type");
  }
}

} // namespace mlir::iree_compiler::IREE::HAL

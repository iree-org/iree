// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/VM/Analysis/RegisterAllocation.h"

#include <algorithm>

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace iree_compiler {

// static
LogicalResult RegisterAllocation::annotateIR(IREE::VM::FuncOp funcOp) {
  RegisterAllocation registerAllocation;
  if (failed(registerAllocation.recalculate(funcOp))) {
    funcOp.emitOpError() << "failed to allocate registers for function";
    return failure();
  }

  Builder builder(funcOp.getContext());
  for (auto &block : funcOp.getBlocks()) {
    SmallVector<Attribute, 8> blockAttrs;
    blockAttrs.reserve(block.getNumArguments());
    for (auto *blockArg : block.getArguments()) {
      uint8_t reg = registerAllocation.map_[blockArg];
      blockAttrs.push_back(builder.getI32IntegerAttr(reg));
    }
    block.front().setAttr("block_registers", builder.getArrayAttr(blockAttrs));

    for (auto &op : block.getOperations()) {
      if (op.getNumResults() == 0) continue;
      SmallVector<Attribute, 8> regAttrs;
      regAttrs.reserve(op.getNumResults());
      for (auto &result : op.getOpResults()) {
        uint8_t reg = registerAllocation.map_[&result];
        regAttrs.push_back(builder.getI32IntegerAttr(reg));
      }
      op.setAttr("result_registers", builder.getArrayAttr(regAttrs));
    }
  }

  return success();
}

// Forms a register reference byte as interpreted by the VM.
// Assumes that the ordinal has been constructed in the valid range.
static uint8_t makeRegisterByte(Type type, int ordinal, bool isMove) {
  if (type.isIntOrIndexOrFloat()) {
    assert(ordinal < kIntRegisterCount);
    return ordinal;
  } else {
    assert(ordinal < kRefRegisterCount);
    return (ordinal | kRefRegisterTypeBit) | (isMove ? kRefRegisterMoveBit : 0);
  }
}

// Bitmaps set indicating which registers of which banks are in use.
struct RegisterUsage {
  llvm::BitVector intRegisters{kIntRegisterCount};
  llvm::BitVector refRegisters{kRefRegisterCount};
  int maxI32RegisterOrdinal = -1;
  int maxRefRegisterOrdinal = -1;

  void reset() {
    intRegisters.reset();
    refRegisters.reset();
    maxI32RegisterOrdinal = -1;
    maxRefRegisterOrdinal = -1;
  }

  Optional<uint8_t> allocateRegister(Type type) {
    if (type.isIntOrIndexOrFloat()) {
      int ordinal = intRegisters.find_first_unset();
      if (ordinal >= kIntRegisterCount) {
        return {};
      }
      intRegisters.set(ordinal);
      maxI32RegisterOrdinal = std::max(ordinal, maxI32RegisterOrdinal);
      return makeRegisterByte(type, ordinal, /*isMove=*/false);
    } else {
      int ordinal = refRegisters.find_first_unset();
      if (ordinal >= kRefRegisterCount) {
        return {};
      }
      refRegisters.set(ordinal);
      maxRefRegisterOrdinal = std::max(ordinal, maxRefRegisterOrdinal);
      return makeRegisterByte(type, ordinal, /*isMove=*/false);
    }
  }

  void releaseRegister(uint8_t reg) {
    if (isRefRegister(reg)) {
      assert(refRegisters.test(reg & 0x3F));
      refRegisters.reset(reg & 0x3F);
    } else {
      assert(intRegisters.test(reg & 0x7F));
      intRegisters.reset(reg & 0x7F);
    }
  }
};

// NOTE: this is not a good algorithm, nor is it a good allocator. If you're
// looking at this and have ideas of how to do this for real please feel
// free to rip it all apart :)
//
// Because I'm lazy we really only look at individual blocks at a time. It'd
// be much better to use dominance info to track values across blocks and
// ensure we are avoiding as many moves as possible. The special case we need to
// handle is when values are not defined within the current block (as values in
// dominators are allowed to cross block boundaries outside of arguments).
LogicalResult RegisterAllocation::recalculate(IREE::VM::FuncOp funcOp) {
  map_.clear();

  if (failed(liveness_.recalculate(funcOp))) {
    return funcOp.emitError()
           << "failed to caclculate required liveness information";
  }

  // Run through each block and allocate registers as we go.
  // We first allocate block arguments and then process each op in-turn to
  // specify their result registers.
  //
  // Note that the entry block arguments must be left-aligned in the register
  // banks as part of the argument passing ABI.
  RegisterUsage registerUsage;
  for (auto &block : funcOp.getBlocks()) {
    for (auto *blockArg : block.getArguments()) {
      auto reg = registerUsage.allocateRegister(blockArg->getType());
      if (!reg.hasValue()) {
        return funcOp.emitError() << "register allocation failed for block arg "
                                  << blockArg->getArgNumber();
      }
      map_[blockArg] = reg.getValue();
      if (blockArg->use_empty()) {
        registerUsage.releaseRegister(reg.getValue());
      }
    }

    for (auto &op : block.getOperations()) {
      for (auto &operand : op.getOpOperands()) {
        if (liveness_.isLastValueUse(operand.get(), &op,
                                     operand.getOperandNumber())) {
          registerUsage.releaseRegister(map_[operand.get()]);
        }
      }
      for (auto &result : op.getOpResults()) {
        auto reg = registerUsage.allocateRegister(result.getType());
        if (!reg.hasValue()) {
          return op.emitError() << "register allocation failed for result "
                                << result.getResultNumber();
        }
        map_[&result] = reg.getValue();
        if (result.use_empty()) {
          registerUsage.releaseRegister(reg.getValue());
        }
      }
    }
  }
  maxI32RegisterOrdinal_ = registerUsage.maxI32RegisterOrdinal;
  maxRefRegisterOrdinal_ = registerUsage.maxRefRegisterOrdinal;

  return success();
}

uint8_t RegisterAllocation::mapToRegister(Value *value) {
  auto it = map_.find(value);
  assert(it != map_.end());
  return it->getSecond();
}

uint8_t RegisterAllocation::mapUseToRegister(Value *value, Operation *useOp,
                                             int operandIndex) {
  uint8_t reg = mapToRegister(value);
  if (isRefRegister(reg) &&
      liveness_.isLastValueUse(value, useOp, operandIndex)) {
    reg |= kRefRegisterMoveBit;
  }
  return reg;
}

}  // namespace iree_compiler
}  // namespace mlir

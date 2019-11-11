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

// IREE high-level sequencer op definitions.
// This op set contains pseudo ops, ops that accept non-MemRef types, and ops in
// normal SSA form.
//
// Through lowering these high-level ops are converted to low-level ops in the
// LLOps.td (iree_ll_seq.*). These map 1:1 with the bytecode, accept
// only MemRef types, and generally use output parameters instead of return
// types.
//
// The source of truth for bytecode opcodes is:
//   https://github.com/google/iree/tree/master/iree/iree/schemas/bytecode/sequencer_bytecode_v0.h

#ifndef IREE_SEQUENCER_HL_OPS
#define IREE_SEQUENCER_HL_OPS

#ifndef IREE_OP_BASE
include "iree/compiler/IR/OpBase.td"
#endif  // IREE_OP_BASE

def IREESeqHL_Dialect : Dialect {
  let name = "iree_hl_seq";
  let cppNamespace = "IREESeq::HL";
}

//===----------------------------------------------------------------------===//
// Base op classes
//===----------------------------------------------------------------------===//

class IREESeqHL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREESeqHL_Dialect, mnemonic, traits>;

class IREESeqHL_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREESeqHL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// High-level sequencer ops
//===----------------------------------------------------------------------===//

def IREESeqHL_CallOp : IREESeqHL_PureOp<"call"> {
  let arguments = (ins SymbolRefAttr:$callee, Variadic<IREEHL_MemRef>);
  let results = (outs Variadic<IREEHL_MemRef>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, FuncOp callee,"
    "ArrayRef<Value *> operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  }]>, OpBuilder<
    "Builder *builder, OperationState &result, StringRef callee,"
    "ArrayRef<Type> results, ArrayRef<Value *> operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(results);
  }]>];

  let extraClassDeclaration = [{
    // TODO(b/132296600): make tablegen follow the style guide.
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    // TODO(b/133879130): make tablegen support variadic operand accessors.
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREESeqHL_CallIndirectOp : IREESeqHL_Op<"call_indirect"> {
  let arguments = (ins FunctionType:$callee, Variadic<IREEHL_MemRef>:$operands);
  let results = (outs Variadic<IREEHL_MemRef>);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *callee,"
    "ArrayRef<Value *> operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addTypes(callee->getType().cast<FunctionType>().getResults());
  }]>];

  let extraClassDeclaration = [{
    // TODO(b/132296600): make tablegen follow the style guide.
    Value *getCallee() { return getOperand(0); }

    // TODO(b/133879130): make tablegen support variadic operand accessors.
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }
  }];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREESeqHL_ReturnOp : IREESeqHL_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<IREEHL_MemRef>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREESeqHL_BranchOp : IREESeqHL_Op<"br", [Terminator]> {
  let arguments = (ins Variadic<IREEHL_MemRef>:$operands);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Block *dest,"
    "ArrayRef<Value *> operands = {}", [{
      result.addSuccessor(dest, operands);
  }]>];

  let extraClassDeclaration = [{
    Block *getDest();
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREESeqHL_CondBranchOp : IREESeqHL_Op<"cond_br", [Terminator]> {
  let arguments = (ins
      IREEHL_BoolScalar:$condition,
      Variadic<IREEHL_MemRef>:$branchOperands
  );

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *condition, "
    "Block *trueDest, ArrayRef<Value *> trueOperands, "
    "Block *falseDest, ArrayRef<Value *> falseOperands", [{
      result.addOperands(condition);
      result.addSuccessor(trueDest, trueOperands);
      result.addSuccessor(falseDest, falseOperands);
  }]>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // The condition operand is the first operand in the list.
    Value *getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getOperation()->getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getOperation()->getSuccessor(falseIndex);
    }

    // Accessors for operands to the 'true' destination.
    Value *getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value *value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_iterator true_operand_begin() {
      return operand_begin() + getTrueDestOperandIndex();
    }
    operand_iterator true_operand_end() {
      return true_operand_begin() + getNumTrueOperands();
    }
    operand_range getTrueOperands() {
      return {true_operand_begin(), true_operand_end()};
    }

    unsigned getNumTrueOperands()  {
      return getOperation()->getNumSuccessorOperands(trueIndex);
    }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getOperation()->eraseSuccessorOperand(trueIndex, index);
    }

    // Accessors for operands to the 'false' destination.
    Value *getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value *value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_iterator false_operand_begin() { return true_operand_end(); }
    operand_iterator false_operand_end() {
      return false_operand_begin() + getNumFalseOperands();
    }
    operand_range getFalseOperands() {
      return {false_operand_begin(), false_operand_end()};
    }

    unsigned getNumFalseOperands() {
      return getOperation()->getNumSuccessorOperands(falseIndex);
    }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getOperation()->eraseSuccessorOperand(falseIndex, index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREESeqHL_DispatchOp : IREESeqHL_Op<"dispatch"> {
  let arguments = (ins
    SymbolRefAttr:$executable,
    SymbolRefAttr:$entry_point,
    IREEHL_IntMemRef:$workload,
    Variadic<IREEHL_MemRef>:$operands
  );
  let results = (outs Variadic<IREEHL_MemRef>);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef executable,"
    "StringRef entry_point, Value *workload,"
    "ArrayRef<Type> results, ArrayRef<Value *> operands = {}", [{
      result.addOperands({workload});
      result.addOperands(operands);
      result.addAttribute("executable", builder->getSymbolRefAttr(executable));
      result.addAttribute("entry_point", builder->getSymbolRefAttr(entry_point));
      result.addTypes(results);
  }]>];

  let extraClassDeclaration = [{
    // TODO(b/132296600): make tablegen follow the style guide.
    StringRef getExecutable() { return executable(); }
    StringRef getEntryPoint() { return entry_point(); }
    FunctionType getEntryPointType();

    Value *getWorkload() { return getOperand(0); }

    // TODO(b/133879130): make tablegen support variadic operand accessors.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin() + 1; }
    operand_iterator arg_operand_end() { return operand_end(); }

    operand_type_range getArgOperandTypes() {
      return {arg_operand_type_begin(), arg_operand_type_end()};
    }
    operand_type_iterator arg_operand_type_begin() {
      return operand_type_iterator(arg_operand_begin());
    }
    operand_type_iterator arg_operand_type_end() {
      return operand_type_iterator(arg_operand_end());
    }
  }];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
  let verifier = [{ return verify$cppClass(*this); }];
}

// TODO(b/142012496): Add trait that enables DCE but not CSE.
def IREESeqHL_AllocHeapOp : IREESeqHL_Op<"alloc_heap"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins Variadic<IREEHL_IntMemRef>:$dim_pieces);
  let results = (outs IREEHL_MemRef);
}

def IREESeqHL_DiscardOp : IREESeqHL_Op<"discard"> {
  let arguments = (ins IREEHL_MemRef);
}

def IREESeqHL_RankOp : IREESeqHL_PureOp<"rank"> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_IntScalar);

  let hasFolder = 1;
}

def IREESeqHL_DimOp : IREESeqHL_PureOp<"dim"> {
  // TODO(benvanik) add dim attr (I32Attr:$dim)
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_IntScalar);
}

def IREESeqHL_ShapeOp : IREESeqHL_PureOp<"shape"> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_1DIntMemRef);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *operand">];

  let hasFolder = 1;
}

def IREESeqHL_LengthOp : IREESeqHL_PureOp<"length"> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_IndexScalar);

  let hasFolder = 1;
}

def IREESeqHL_SliceOp :
    IREESeqHL_PureOp<"slice", [AllElementTypesMatch<["src", "result"]>,
                               AllTypesMatch<["indices", "lengths"]>]> {
  let arguments = (ins
      IREEHL_MemRef:$src,
      IREEHL_1DIndexMemRef:$indices,
      IREEHL_1DIndexMemRef:$lengths
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREESeqHL_CopyOp : IREESeqHL_Op<"copy", [
  AllElementCountsMatch<["srcIndices", "dstIndices", "lengths"]>,
  AllRanksMatch<["src", "dst"]>,
  // The checks above are redundant with this one, but they give more specific
  // error messages.
  AllMatch<[
    Rank<"src">.result,
    Rank<"dst">.result,
    ElementCount<"srcIndices">.result,
    ElementCount<"dstIndices">.result,
    ElementCount<"lengths">.result
  ], "src/dst rank is the same as srcIndices/dstIndices/lengths size">,
  AllElementTypesMatch<["src", "dst"]>
]> {
  let arguments = (ins
    IREEHL_MemRef:$src,
    IREEHL_1DIndexMemRef:$srcIndices,
    IREEHL_MemRef:$dst,
    IREEHL_1DIndexMemRef:$dstIndices,
    IREEHL_1DIndexMemRef:$lengths
  );
}

def IREESeqHL_FillOp : IREESeqHL_Op<"fill"> {
  let arguments = (ins
    IREEHL_I32Scalar:$value,
    IREEHL_MemRef:$dst,
    IREEHL_1DIndexMemRef:$dstIndices,
    IREEHL_1DIndexMemRef:$lengths
  );
}

def IREESeqHL_CloneOp : IREESeqHL_PureOp<"clone", [SameOperandsAndResultType]> {
  let arguments = (ins IREEHL_MemRef:$src);
  let results = (outs IREEHL_MemRef);
}

// A pseudo op provided for convenience. This gets canonicalized to a series of
// copies.
def IREESeqHL_ConcatOp : IREESeqHL_PureOp<"concat"> {
  // TODO(b/135032064) Add type constraints when they support variadic
  let arguments = (ins
      Variadic<IREEHL_MemRef>:$srcs,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_MemRef);

  let hasCanonicalizer = 1;
}

def IREESeqHL_AssignOp :
    IREESeqHL_PureOp<"assign", [SameOperandsAndResultType]> {
  let arguments = (ins IREEHL_MemRef:$src);
  let results = (outs IREEHL_MemRef);
}

def IREESeqHL_CondAssignOp : IREESeqHL_PureOp<"cond_assign"> {
  let arguments = (ins
      IREEHL_BoolScalar:$cond,
      IREEHL_MemRef:$lhs,
      IREEHL_MemRef:$rhs
  );
  let results = (outs IREEHL_MemRef);
}

def IREESeqHL_ReshapeOp : IREESeqHL_PureOp<"reshape"> {
  let arguments = (ins IREEHL_MemRef:$src, IREEHL_MemRef:$shape);
  let results = (outs IREEHL_MemRef);
}

def IREESeqHL_TraceOp : IREESeqHL_Op<"trace"> {
  let arguments = (ins Variadic<IREEHL_MemRef>:$srcs);
}

def IREESeqHL_CondBreakOp : IREESeqHL_Op<"cond_break"> {
  let arguments = (ins IREEHL_BoolScalar:$cond);
}

def IREESeqHL_BreakOp : IREESeqHL_Op<"break">;

#endif  // IREE_SEQUENCER_HL_OPS

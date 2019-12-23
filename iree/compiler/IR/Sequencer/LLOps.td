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

// IREE low-level sequencer op definitions.
// These map 1:1 with the bytecode, accept only MemRef types and generally use
// output parameters instead of return types.
//
// The source of truth for bytecode opcodes is:
//   https://github.com/google/iree/tree/master/iree/iree/schemas/bytecode/sequencer_bytecode_v0.h
//
// Note that in this dialect we cannot use folders: they require that all
// operands are possible to make constants where we use output arguments that
// will never be constant. Instead we can use canonicalization patterns to
// match constant input operands and do the folding by replacing output operands
// with the new values.

#ifndef IREE_SEQUENCER_LL_OPS
#define IREE_SEQUENCER_LL_OPS

#ifndef IREE_OP_BASE
include "iree/compiler/IR/OpBase.td"
#endif  // IREE_OP_BASE

def IREESeqLL_Dialect : Dialect {
  let name = "iree_ll_seq";
  let cppNamespace = "IREESeq::LL";
}

//===----------------------------------------------------------------------===//
// Base op classes
//===----------------------------------------------------------------------===//

class IREESeqLL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREESeqLL_Dialect, mnemonic, traits> {
  bit hasCustomSerializer = 0;
}

class IREESeqLL_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREESeqLL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

class IREESeqLL_UnaryOp<string mnemonic, Type type = IREELL_MemRef,
                     list<OpTrait> traits = []> : IREESeqLL_Op<mnemonic, traits> {
  let arguments = (ins type:$input, type:$dst);
}

class IREESeqLL_BinaryOp<string mnemonic, Type type = IREELL_MemRef,
                      list<OpTrait> traits = []> : IREESeqLL_Op<mnemonic, traits> {
  let arguments = (ins type:$lhs, type:$rhs, type:$dst);
}

class IREESeqLL_TernaryOp<string mnemonic, Type type = IREELL_MemRef,
                       list<OpTrait> traits = []>
    : IREESeqLL_Op<mnemonic, traits> {
  let arguments = (ins type : $a, type : $b, type : $c, type : $dst);
}

//===----------------------------------------------------------------------===//
// Low-level sequencer ops
//===----------------------------------------------------------------------===//

def IREESeqLL_ConstantOp : IREESeqLL_PureOp<"constant"> {
  let arguments = (ins ElementsAttr:$value);
  let results = (outs IREELL_MemRef);

  // TODO(b/132296600): make tablegen follow the style guide.
  let extraClassDeclaration = [{
    Attribute getValue() { return value(); }
  }];

  let hasFolder = 1;
}

def IREESeqLL_CallOp : IREESeqLL_Op<"call"> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<IREELL_MemRef>);
  let results = (outs Variadic<IREELL_MemRef>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, FuncOp callee,"
    "ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  }]>, OpBuilder<
    "Builder *builder, OperationState &result, StringRef callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
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

// TODO(benvanik): add verifier that target isExternal.
def IREESeqLL_CallImportOp : IREESeqLL_Op<"call_import"> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<IREELL_MemRef>);
  let results = (outs Variadic<IREELL_MemRef>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, FuncOp callee,"
    "ValueRange operands = {}", [{
      result.addOperands(operands);
      result.addAttribute("callee", builder->getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
  }]>, OpBuilder<
    "Builder *builder, OperationState &result, StringRef callee,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
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

def IREESeqLL_CallIndirectOp : IREESeqLL_Op<"call_indirect"> {
  let arguments = (ins FunctionType:$callee, Variadic<IREELL_MemRef>:$operands);
  let results = (outs Variadic<IREELL_MemRef>);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value callee,"
    "ValueRange operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addTypes(callee->getType().cast<FunctionType>().getResults());
  }]>];

  let extraClassDeclaration = [{
    Value getCallee() { return getOperand(0); }

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

def IREESeqLL_ReturnOp : IREESeqLL_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<IREELL_MemRef>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREESeqLL_BranchOp : IREESeqLL_Op<"br", [Terminator]> {
  let arguments = (ins Variadic<IREELL_MemRef>:$operands);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Block *dest, "
    "ValueRange operands = {}", [{
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

def IREESeqLL_CondBranchOp : IREESeqLL_Op<"cond_br", [Terminator]> {
  let arguments = (ins
      IREELL_BoolScalar:$condition,
      Variadic<IREELL_MemRef>:$branchOperands
  );

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value condition,"
    "Block *trueDest, ValueRange trueOperands,"
    "Block *falseDest, ValueRange falseOperands", [{
      result.addOperands(condition);
      result.addSuccessor(trueDest, trueOperands);
      result.addSuccessor(falseDest, falseOperands);
  }]>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // The condition operand is the first operand in the list.
    Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getOperation()->getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getOperation()->getSuccessor(falseIndex);
    }

    // Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
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
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
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

def IREESeqLL_DynamicDispatchOp : IREESeqLL_Op<"dynamic_dispatch"> {
  let arguments = (ins
    FlatSymbolRefAttr:$executable,
    FlatSymbolRefAttr:$entry_point,
    IREELL_IntMemRef:$workload,
    Variadic<IREELL_MemRef>:$operands
  );
  let results = (outs Variadic<IREELL_MemRef>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef executable,"
    "StringRef entry_point, Value workload,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
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

    Value getWorkload() { return getOperand(0); }

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
  let hasCanonicalizer = 1;
}

def IREESeqLL_StaticDispatchOp : IREESeqLL_Op<"static_dispatch"> {
  let arguments = (ins
    FlatSymbolRefAttr:$executable,
    FlatSymbolRefAttr:$entry_point,
    I32ElementsAttr:$workload,
    Variadic<IREELL_MemRef>:$operands
  );
  let results = (outs Variadic<IREELL_MemRef>);

  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, StringRef executable,"
    "StringRef entry_point, ElementsAttr workload,"
    "ArrayRef<Type> results, ValueRange operands = {}", [{
      result.addAttribute("workload", workload);
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

    ElementsAttr getWorkload() { return workload(); }

    // TODO(b/133879130): make tablegen support variadic operand accessors.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
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

def IREESeqLL_AllocStaticOp : IREESeqLL_PureOp<"alloc_static"> {
  // TODO(benvanik): attributes and args.
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_AllocStackOp : IREESeqLL_PureOp<"alloc_stack"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins Variadic<IREELL_IntMemRef>:$dim_pieces);
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_AllocStackInitOp : IREESeqLL_PureOp<"alloc_stack_init"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins Variadic<IREELL_IntMemRef>:$dim_pieces);
  let results = (outs IREELL_MemRef);
}

// TODO(b/142012496): Add trait that enables DCE but not CSE.
def IREESeqLL_AllocHeapOp : IREESeqLL_Op<"alloc_heap"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins Variadic<IREELL_IntMemRef>:$dim_pieces);
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_DiscardOp : IREESeqLL_Op<"discard"> {
  let arguments = (ins IREELL_MemRef);
}

def IREESeqLL_ShapeOp : IREESeqLL_Op<"shape"> {
  let arguments = (ins IREELL_MemRef:$input, IREELL_I32MemRef:$dst);

  let hasCanonicalizer = 1;
}

def IREESeqLL_LengthOp : IREESeqLL_Op<"length"> {
  let arguments = (ins IREELL_MemRef:$input, IREELL_I32Scalar:$dst);

  let hasCanonicalizer = 1;
}

def IREESeqLL_ComputeOffsetOp : IREESeqLL_Op<"compute_offset"> {
  let arguments = (ins
      IREELL_1DIntMemRef:$shape,
      I8Attr:$elementSize,
      IREELL_1DIntMemRef:$indices,
      IREELL_I32Scalar:$dst
  );

  let hasCanonicalizer = 1;
}

def IREESeqLL_ComputeRangeOp : IREESeqLL_Op<"compute_range"> {
  let arguments = (ins
      IREELL_1DIntMemRef:$shape,
      I8Attr:$elementSize,
      IREELL_1DIntMemRef:$indices,
      IREELL_1DIntMemRef:$lengths,
      IREELL_I32Scalar:$dstOffset,
      IREELL_I32Scalar:$dstLength
  );

  let hasCanonicalizer = 1;
}

def IREESeqLL_DynamicSliceOp : IREESeqLL_PureOp<"dynamic_slice", [
  AllElementTypesMatch<["src", "result"]>
]> {
  let arguments = (ins
      IREELL_MemRef:$src,
      IREELL_IntScalar:$offset,
      IREELL_IntScalar:$length
  );
  let results = (outs IREELL_MemRef:$result);
}

def IREESeqLL_StaticSliceOp : IREESeqLL_PureOp<"static_slice", [
  AllElementTypesMatch<["src", "result"]>
]> {
  let arguments = (ins
      IREELL_MemRef:$src,
      I64Attr:$offset,
      I64Attr:$length
  );
  let results = (outs IREELL_MemRef:$result);
}

def IREESeqLL_DynamicCopyOp : IREESeqLL_Op<"dynamic_copy"> {
  let arguments = (ins
      IREELL_MemRef:$src,
      IREELL_IndexScalar:$srcOffset,
      IREELL_MemRef:$dst,
      IREELL_IndexScalar:$dstOffset,
      IREELL_IndexScalar:$length
  );

  let hasCanonicalizer = 1;
}

def IREESeqLL_StaticCopyOp : IREESeqLL_Op<"static_copy"> {
  let arguments = (ins
      IREELL_MemRef:$src,
      I64Attr:$srcOffset,
      IREELL_MemRef:$dst,
      I64Attr:$dstOffset,
      I64Attr:$length
  );
}

def IREESeqLL_DynamicFillOp : IREESeqLL_Op<"dynamic_fill"> {
  let arguments = (ins
      IREELL_I32Scalar:$value,
      IREELL_MemRef:$dst,
      IREELL_IndexScalar:$dstOffset,
      IREELL_IndexScalar:$length
  );

  let hasCanonicalizer = 1;
}

def IREESeqLL_StaticFillOp : IREESeqLL_Op<"static_fill"> {
  let arguments = (ins
      I32Attr:$value,
      IREELL_MemRef:$dst,
      I64Attr:$dstOffset,
      I64Attr:$length
  );
}

def IREESeqLL_CloneOp :
    IREESeqLL_PureOp<"clone", [SameOperandsAndResultType]> {
  let arguments = (ins IREELL_MemRef:$src);
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_AssignOp :
    IREESeqLL_Op<"assign", [SameOperandsAndResultType]> {
  let arguments = (ins IREELL_MemRef:$src);
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_CondAssignOp : IREESeqLL_Op<"cond_assign"> {
  let arguments = (ins
      IREELL_BoolScalar:$cond,
      IREELL_MemRef:$lhs,
      IREELL_MemRef:$rhs
  );
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_ReshapeOp : IREESeqLL_Op<"reshape"> {
  let arguments = (ins IREELL_MemRef:$input, IREELL_1DIntMemRef:$shape);
  let results = (outs IREELL_MemRef);
}

def IREESeqLL_TraceOp : IREESeqLL_Op<"trace"> {
  let arguments = (ins Variadic<IREELL_MemRef>:$srcs);
}

def IREESeqLL_CondBreakOp : IREESeqLL_Op<"cond_break"> {
  let arguments = (ins IREELL_BoolScalar:$cond);
}

def IREESeqLL_BreakOp : IREESeqLL_Op<"break">;

#endif  // IREE_SEQUENCER_LL_OPS

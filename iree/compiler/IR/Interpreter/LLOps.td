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

// IREE low-level interpreter op definitions.
// These map 1:1 with the bytecode, accept only MemRef types and generally use
// output parameters instead of return types.
//
// The source of truth for bytecode opcodes is:
//   https://github.com/google/iree/tree/master/iree/iree/schemas/bytecode/interpreter_bytecode_v0.h

#ifndef IREE_INTERPRETER_LL_OPS
#define IREE_INTERPRETER_LL_OPS

#ifndef IREE_OP_BASE
include "iree/compiler/IR/OpBase.td"
#endif  // IREE_OP_BASE

def IREEInterpLL_Dialect : Dialect {
  let name = "iree_ll_interp";
  let cppNamespace = "IREEInterp::LL";
}

//===----------------------------------------------------------------------===//
// Base op classes
//===----------------------------------------------------------------------===//

class IREEInterpLL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREEInterpLL_Dialect, mnemonic, traits>;

class IREEInterpLL_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREEInterpLL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

class IREEInterpLL_UnaryOp<string mnemonic, Type type = IREELL_MemRef,
                     list<OpTrait> traits = []> : IREEInterpLL_Op<mnemonic, traits> {
  let arguments = (ins type:$input, type:$dst);
}

class IREEInterpLL_BinaryOp<string mnemonic, Type type = IREELL_MemRef,
                      list<OpTrait> traits = []> : IREEInterpLL_Op<mnemonic, traits> {
  let arguments = (ins type:$lhs, type:$rhs, type:$dst);
}

class IREEInterpLL_TernaryOp<string mnemonic, Type type = IREELL_MemRef,
                       list<OpTrait> traits = []>
    : IREEInterpLL_Op<mnemonic, traits> {
  let arguments = (ins type : $a, type : $b, type : $c, type : $dst);
}

//===----------------------------------------------------------------------===//
// Low-level interpreter ops
//===----------------------------------------------------------------------===//

// TODO(benvanik): value attribute.
def IREEInterpLL_ConstantOp : IREEInterpLL_PureOp<"constant"> {
  let results = (outs IREELL_MemRef);
}

def IREEInterpLL_CallOp : IREEInterpLL_Op<"call"> {
  let arguments = (ins SymbolRefAttr:$callee, Variadic<IREELL_MemRef>);
  let results = (outs Variadic<IREELL_MemRef>);

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

// TODO(benvanik): add verifier that target isExternal.
def IREEInterpLL_CallImportOp : IREEInterpLL_Op<"call_import"> {
  let arguments = (ins SymbolRefAttr:$callee, Variadic<IREELL_MemRef>);
  let results = (outs Variadic<IREELL_MemRef>);

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

def IREEInterpLL_CallIndirectOp : IREEInterpLL_Op<"call_indirect"> {
  let arguments = (ins FunctionType:$callee, Variadic<IREELL_MemRef>:$operands);
  let results = (outs Variadic<IREELL_MemRef>);

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *callee,"
    "ArrayRef<Value *> operands = {}", [{
      result.operands.push_back(callee);
      result.addOperands(operands);
      result.addTypes(callee->getType().cast<FunctionType>().getResults());
  }]>];

  let extraClassDeclaration = [{
    Value *getCallee() { return getOperand(0); }

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

def IREEInterpLL_ReturnOp : IREEInterpLL_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<IREELL_MemRef>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREEInterpLL_BranchOp : IREEInterpLL_Op<"br", [Terminator]> {
  let arguments = (ins Variadic<IREELL_MemRef>:$operands);

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

def IREEInterpLL_CondBranchOp : IREEInterpLL_Op<"cond_br", [Terminator]> {
  let arguments = (ins
      IREELL_BoolScalar:$condition,
      Variadic<IREELL_MemRef>:$branchOperands
  );

  let builders = [OpBuilder<
    "Builder *, OperationState &result, Value *condition,"
    "Block *trueDest, ArrayRef<Value *> trueOperands,"
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

def IREEInterpLL_CmpIOp : IREEInterpLL_Op<"cmp_i"> {
  let arguments = (ins
      I32Attr:$predicate,
      IREELL_IntMemRef:$lhs,
      IREELL_IntMemRef:$rhs,
      IREELL_BoolMemRef:$dst
  );
}

def IREEInterpLL_CmpFOp : IREEInterpLL_Op<"cmp_f"> {
  let arguments = (ins
      I32Attr:$predicate,
      IREELL_FloatMemRef:$lhs,
      IREELL_FloatMemRef:$rhs,
      IREELL_BoolMemRef:$dst
  );
}

def IREEInterpLL_AllocStaticOp : IREEInterpLL_PureOp<"alloc_static"> {
  // TODO(benvanik): attributes and args.
  let results = (outs IREELL_MemRef);
}

def IREEInterpLL_AllocStackOp : IREEInterpLL_PureOp<"alloc_stack"> {
  // TODO(benvanik): atributes and args.
  let arguments = (ins
      Variadic<IREELL_MemRef>:$dim_pieces
  );
  let results = (outs
      IREELL_MemRef
  );
}

def IREEInterpLL_AllocStackInitOp : IREEInterpLL_PureOp<"alloc_stack_init"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins
      Variadic<IREELL_MemRef>:$dim_pieces
  );
  let results = (outs
      IREELL_MemRef
  );
}

// TODO(b/142012496): Add trait that enables DCE but not CSE.
def IREEInterpLL_AllocHeapOp : IREEInterpLL_Op<"alloc_heap"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins
      Variadic<IREELL_MemRef>:$dim_pieces
  );
  let results = (outs
      IREELL_MemRef
  );
}

def IREEInterpLL_DiscardOp : IREEInterpLL_Op<"discard"> {
  let arguments = (ins IREELL_MemRef);
}

def IREEInterpLL_RankOp : IREEInterpLL_Op<"rank"> {
  let arguments = (ins
      IREELL_MemRef:$input,
      IREELL_I32Scalar:$dst
  );
}

def IREEInterpLL_DimOp : IREEInterpLL_Op<"dim"> {
  // TODO(benvanik) add dim attr (I32Attr:$dim)
  let arguments = (ins
      IREELL_MemRef:$input,
      IREELL_I32Scalar:$dst
  );
}

def IREEInterpLL_ShapeOp : IREEInterpLL_Op<"shape"> {
  let arguments = (ins
      IREELL_MemRef:$input,
      IREELL_I32MemRef:$dst
  );
}

def IREEInterpLL_LengthOp : IREEInterpLL_Op<"length"> {
  let arguments = (ins
      IREELL_MemRef:$input,
      IREELL_I32Scalar:$dst
  );
}


def IREEInterpLL_DynamicSliceOp : IREEInterpLL_PureOp<"dynamic_slice"> {
  let arguments = (ins
      IREELL_MemRef:$src,
      IREELL_1DIndexMemRef:$srcIndices,
      IREELL_1DIndexMemRef:$lengths
  );
  let results = (outs
      IREELL_MemRef
  );
}

// TODO(benvanik): add attribute requirements/types.
def IREEInterpLL_StaticSliceOp :
    IREEInterpLL_PureOp<"static_slice", [SameOperandsAndResultElementType]> {
  let arguments = (ins IREELL_MemRef:$src);
  let results = (outs IREELL_MemRef);
}

def IREEInterpLL_DynamicCopyOp : IREEInterpLL_Op<"dynamic_copy", [
  AllElementCountsMatch<["srcIndices", "dstIndices", "lengths"]>,
]> {
  let arguments = (ins
      IREELL_MemRef:$src,
      IREELL_1DIndexMemRef:$srcIndices,
      IREELL_MemRef:$dst,
      IREELL_1DIndexMemRef:$dstIndices,
      IREELL_1DIndexMemRef:$lengths
  );
}

def IREEInterpLL_StaticCopyOp : IREEInterpLL_Op<"static_copy", [
  AllElementCountsMatch<["srcIndices", "dstIndices", "lengths"]>,
]> {
  let arguments = (ins
      IREELL_MemRef:$src,
      I32ElementsAttr:$srcIndices,
      IREELL_MemRef:$dst,
      I32ElementsAttr:$dstIndices,
      I32ElementsAttr:$lengths
  );
}

def IREEInterpLL_CloneOp :
    IREEInterpLL_PureOp<"clone", [SameOperandsAndResultType]> {
  let arguments = (ins IREELL_MemRef:$src);
  let results = (outs IREELL_MemRef);
}

// TODO(benvanik): add split dim/size/etc. Maybe make multiple ops?
def IREEInterpLL_SplitOp : IREEInterpLL_PureOp<"split"> {
  let arguments = (ins
      IREELL_MemRef:$src
  );
  let results = (outs
      Variadic<IREELL_MemRef>
  );
}

def IREEInterpLL_AssignOp :
    IREEInterpLL_Op<"assign", [SameOperandsAndResultType]> {
  let arguments = (ins IREELL_MemRef:$src);
  let results = (outs IREELL_MemRef);
}

def IREEInterpLL_CondAssignOp : IREEInterpLL_Op<"cond_assign"> {
  let arguments = (ins
      IREELL_BoolScalar:$cond,
      IREELL_MemRef:$lhs,
      IREELL_MemRef:$rhs
  );
  let results = (outs
      IREELL_MemRef
  );
}

def IREEInterpLL_ReshapeOp : IREEInterpLL_Op<"reshape"> {
  let arguments = (ins
      IREELL_MemRef:$input,
      IREELL_1DIntMemRef:$shape
  );
  let results = (outs
      IREELL_MemRef
  );
}

def IREEInterpLL_SelectOp : IREEInterpLL_Op<"select"> {
  let arguments = (ins
      IREELL_MemRef:$cond,
      IREELL_MemRef:$lhs,
      IREELL_MemRef:$rhs,
      IREELL_MemRef:$dst
  );
}

def IREEInterpLL_PadOp :
    IREEInterpLL_Op<
        "pad", [AllElementTypesMatch<["src", "dst", "padding_value"]>]> {
  let arguments = (ins
      IREELL_MemRef:$src,
      IREELL_ElementScalar:$padding_value,
      IREELL_1DIndexMemRef:$edge_padding_low,
      IREELL_1DIndexMemRef:$edge_padding_high,
      IREELL_1DIndexMemRef:$interior_padding,
      IREELL_MemRef:$dst
  );
}

def IREEInterpLL_TransposeOp : IREEInterpLL_BinaryOp<"transpose">;

def IREEInterPLL_ReverseOp : IREEInterpLL_BinaryOp<"reverse">;

def IREEInterpLL_BroadcastOp : IREEInterpLL_BinaryOp<"broadcast">;

def IREEInterpLL_TileOp : IREEInterpLL_BinaryOp<"tile">;

// TODO(benvanik): add traits for broadcasting support.

def IREEInterpLL_NotOp : IREEInterpLL_UnaryOp<"not">;
def IREEInterpLL_AndOp : IREEInterpLL_BinaryOp<"and">;
def IREEInterpLL_OrOp : IREEInterpLL_BinaryOp<"or">;
def IREEInterpLL_XorOp : IREEInterpLL_BinaryOp<"xor">;
def IREEInterpLL_ShiftLeftOp : IREEInterpLL_BinaryOp<"sll">;
def IREEInterpLL_ShiftRightLogicalOp : IREEInterpLL_BinaryOp<"srl">;
def IREEInterpLL_ShiftRightArithmeticOp : IREEInterpLL_BinaryOp<"sra">;

def IREEInterpLL_AddIOp : IREEInterpLL_BinaryOp<"add_i", IREELL_IntMemRef>;
def IREEInterpLL_AddFOp : IREEInterpLL_BinaryOp<"add_f", IREELL_FloatMemRef>;
def IREEInterpLL_SubIOp : IREEInterpLL_BinaryOp<"sub_i", IREELL_IntMemRef>;
def IREEInterpLL_SubFOp : IREEInterpLL_BinaryOp<"sub_f", IREELL_FloatMemRef>;
def IREEInterpLL_AbsIOp : IREEInterpLL_UnaryOp<"abs_i", IREELL_IntMemRef>;
def IREEInterpLL_AbsFOp : IREEInterpLL_UnaryOp<"abs_f", IREELL_FloatMemRef>;
def IREEInterpLL_MulIOp : IREEInterpLL_BinaryOp<"mul_i", IREELL_IntMemRef>;
def IREEInterpLL_MulFOp : IREEInterpLL_BinaryOp<"mul_f", IREELL_FloatMemRef>;
def IREEInterpLL_DivISOp : IREEInterpLL_BinaryOp<"div_i_s", IREELL_IntMemRef>;
def IREEInterpLL_DivIUOp : IREEInterpLL_BinaryOp<"div_i_u", IREELL_IntMemRef>;
def IREEInterpLL_DivFOp : IREEInterpLL_BinaryOp<"div_f", IREELL_FloatMemRef>;
def IREEInterpLL_MulAddIOp : IREEInterpLL_BinaryOp<"madd_i", IREELL_IntMemRef>;
def IREEInterpLL_MulAddFOp : IREEInterpLL_BinaryOp<"madd_f", IREELL_FloatMemRef>;
def IREEInterpLL_ExpFOp : IREEInterpLL_UnaryOp<"exp_f", IREELL_FloatMemRef>;
def IREEInterpLL_LogFOp : IREEInterpLL_UnaryOp<"log_f", IREELL_FloatMemRef>;
def IREEInterpLL_RsqrtFOp : IREEInterpLL_UnaryOp<"rsqrt_f", IREELL_FloatMemRef>;
def IREEInterpLL_CosFOp : IREEInterpLL_UnaryOp<"cos_f", IREELL_FloatMemRef>;
def IREEInterpLL_SinFOp : IREEInterpLL_UnaryOp<"sin_f", IREELL_FloatMemRef>;
def IREEInterpLL_TanhFOp : IREEInterpLL_UnaryOp<"tanh_f", IREELL_FloatMemRef>;
def IREEInterpLL_Atan2FOp : IREEInterpLL_UnaryOp<"atan2_f", IREELL_FloatMemRef>;

def IREEInterpLL_MinISOp : IREEInterpLL_BinaryOp<"min_i_s", IREELL_IntMemRef>;
def IREEInterpLL_MinIUOp : IREEInterpLL_BinaryOp<"min_i_u", IREELL_IntMemRef>;
def IREEInterpLL_MinFOp : IREEInterpLL_BinaryOp<"min_f", IREELL_FloatMemRef>;
def IREEInterpLL_MaxISOp : IREEInterpLL_BinaryOp<"max_i_s", IREELL_IntMemRef>;
def IREEInterpLL_MaxIUOp : IREEInterpLL_BinaryOp<"max_i_u", IREELL_IntMemRef>;
def IREEInterpLL_MaxFOp : IREEInterpLL_BinaryOp<"max_f", IREELL_FloatMemRef>;
def IREEInterpLL_ClampFOp : IREEInterpLL_TernaryOp<"clamp_f", IREELL_FloatMemRef>;
def IREEInterpLL_FloorFOp : IREEInterpLL_UnaryOp<"floor_f", IREELL_FloatMemRef>;
def IREEInterpLL_CeilFOp : IREEInterpLL_UnaryOp<"ceil_f", IREELL_FloatMemRef>;

def IREEInterpLL_ConvertSSOp : IREEInterpLL_UnaryOp<"convert_s_s", IREELL_MemRef>;
def IREEInterpLL_ConvertSUOp : IREEInterpLL_UnaryOp<"convert_s_u", IREELL_MemRef>;
def IREEInterpLL_ConvertSFOp : IREEInterpLL_UnaryOp<"convert_s_f", IREELL_MemRef>;

def IREEInterpLL_ConvertUSOp : IREEInterpLL_UnaryOp<"convert_u_s", IREELL_MemRef>;
def IREEInterpLL_ConvertUUOp : IREEInterpLL_UnaryOp<"convert_u_u", IREELL_MemRef>;
def IREEInterpLL_ConvertUFOp : IREEInterpLL_UnaryOp<"convert_u_f", IREELL_MemRef>;

def IREEInterpLL_ConvertFSOp : IREEInterpLL_UnaryOp<"convert_f_s", IREELL_MemRef>;
def IREEInterpLL_ConvertFUOp : IREEInterpLL_UnaryOp<"convert_f_u", IREELL_MemRef>;
def IREEInterpLL_ConvertFFOp : IREEInterpLL_UnaryOp<"convert_f_f", IREELL_MemRef>;

def IREEInterpLL_MatMulIOp : IREEInterpLL_Op<"matmul_i"> {
  let arguments = (ins
      IREELL_IntMemRef:$lhs,
      IREELL_IntMemRef:$rhs,
      IREELL_IntMemRef:$multiplier_mantissa,
      IREELL_IntMemRef:$multiplier_exponent,
      IREELL_IntMemRef:$dst
  );
}
def IREEInterpLL_MatMulFOp : IREEInterpLL_Op<"matmul_f"> {
  let arguments = (ins
      IREELL_FloatMemRef:$lhs,
      IREELL_FloatMemRef:$rhs,
      IREELL_FloatMemRef:$dst
  );
}

def IREEInterpLL_ReduceSumIOp : IREEInterpLL_Op<"reduce_sum_i"> {
  let arguments = (ins
      IREELL_IntMemRef:$src,
      IREELL_IntMemRef:$init,
      I32Attr:$dimension,
      IREELL_IntMemRef:$dst
  );
}
def IREEInterpLL_ReduceSumFOp : IREEInterpLL_Op<"reduce_sum_f"> {
  let arguments = (ins
      IREELL_FloatMemRef:$src,
      IREELL_FloatMemRef:$init,
      I32Attr:$dimension,
      IREELL_FloatMemRef:$dst
  );
}

def IREEInterpLL_ReduceMinIOp : IREEInterpLL_Op<"reduce_min_i"> {
  let arguments = (ins
      IREELL_IntMemRef:$src,
      IREELL_IntMemRef:$init,
      I32Attr:$dimension,
      IREELL_IntMemRef:$dst
  );
}
def IREEInterpLL_ReduceMinFOp : IREEInterpLL_Op<"reduce_min_f"> {
  let arguments = (ins
      IREELL_FloatMemRef:$src,
      IREELL_FloatMemRef:$init,
      I32Attr:$dimension,
      IREELL_FloatMemRef:$dst
  );
}

def IREEInterpLL_ReduceMaxIOp : IREEInterpLL_Op<"reduce_max_i"> {
  let arguments = (ins
      IREELL_IntMemRef:$src,
      IREELL_IntMemRef:$init,
      I32Attr:$dimension,
      IREELL_IntMemRef:$dst
  );
}
def IREEInterpLL_ReduceMaxFOp : IREEInterpLL_Op<"reduce_max_f"> {
  let arguments = (ins
      IREELL_FloatMemRef:$src,
      IREELL_FloatMemRef:$init,
      I32Attr:$dimension,
      IREELL_FloatMemRef:$dst
  );
}

def IREEInterpLL_TraceOp : IREEInterpLL_Op<"trace"> {
  let arguments = (ins
      Variadic<IREELL_MemRef>:$srcs
  );
}

def IREEInterpLL_CondBreakOp : IREEInterpLL_Op<"cond_break"> {
  let arguments = (ins
      IREELL_BoolScalar:$cond
  );
}

def IREEInterpLL_BreakOp : IREEInterpLL_Op<"break">;

#endif  // IREE_INTERPRETER_LL_OPS

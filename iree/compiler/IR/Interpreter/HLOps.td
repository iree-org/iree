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

// IREE high-level interpreter op definitions.
// This op set contains pseudo ops, ops that accept non-MemRef types, and ops in
// normal SSA form.
//
// Through lowering these high-level ops are converted to low-level ops in the
// LLOps.td (iree_ll_interp.*). These map 1:1 with the bytecode,
// accept only MemRef types, and generally use output parameters instead of
// return types.
//
// The source of truth for bytecode opcodes is:
//   https://github.com/google/iree/tree/master/iree/iree/schemas/bytecode/interpreter_bytecode_v0.h

#ifndef IREE_INTERPRETER_HL_OPS
#define IREE_INTERPRETER_HL_OPS

include "iree/compiler/IR/OpBase.td"

def IREEInterpHL_Dialect : Dialect {
  let name = "iree_hl_interp";
  let cppNamespace = "IREEInterp::HL";
}

//===----------------------------------------------------------------------===//
// Base op classes
//===----------------------------------------------------------------------===//

class IREEInterpHL_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREEInterpHL_Dialect, mnemonic, traits>;

class IREEInterpHL_PureOp<string mnemonic, list<OpTrait> traits = []> :
    IREEInterpHL_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

//===----------------------------------------------------------------------===//
// High-level interpreter ops
//===----------------------------------------------------------------------===//

def IREEInterpHL_CallOp : IREEInterpHL_Op<"call"> {
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<IREEHL_MemRef>);
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

def IREEInterpHL_CallIndirectOp : IREEInterpHL_Op<"call_indirect"> {
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

def IREEInterpHL_ReturnOp : IREEInterpHL_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<IREEHL_MemRef>:$operands);

  let builders = [OpBuilder<
    "Builder *b, OperationState &result", [{ build(b, result, llvm::None); }]
  >];

  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ return print$cppClass(p, *this); }];
}

def IREEInterpHL_BranchOp : IREEInterpHL_Op<"br", [Terminator]> {
  let arguments = (ins Variadic<IREEHL_MemRef>:$operands);

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

def IREEInterpHL_CondBranchOp : IREEInterpHL_Op<"cond_br", [Terminator]> {
  let arguments = (ins
      IREEHL_BoolScalar:$condition,
      Variadic<IREEHL_MemRef>:$branchOperands
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

def IREEInterpHL_CmpIOp :
    IREEInterpHL_PureOp<"cmp_i", [SameOperandsAndResultShape,
                                  AllTypesMatch<["lhs", "rhs"]>]> {
  let arguments = (ins
      I32Attr:$predicate,
      IREEHL_IntMemRef:$lhs,
      IREEHL_IntMemRef:$rhs
  );
  let results = (outs IREEHL_BoolMemRef);
}

def IREEInterpHL_CmpFOp :
    IREEInterpHL_PureOp<"cmp_f", [SameOperandsAndResultShape,
                                  AllTypesMatch<["lhs", "rhs"]>]> {
  let arguments = (ins
      I32Attr:$predicate,
      IREEHL_FloatMemRef:$lhs,
      IREEHL_FloatMemRef:$rhs
  );
  let results = (outs IREEHL_BoolMemRef);
}

// TODO(b/142012496): Add trait that enables DCE but not CSE.
def IREEInterpHL_AllocHeapOp : IREEInterpHL_Op<"alloc_heap"> {
  // TODO(benvanik): attributes and args.
  let arguments = (ins
      Variadic<IREEHL_MemRef>:$dim_pieces
  );
  let results = (outs
      IREEHL_MemRef
  );
}

def IREEInterpHL_DiscardOp : IREEInterpHL_Op<"discard"> {
  let arguments = (ins IREEHL_MemRef);
}

def IREEInterpHL_RankOp : IREEInterpHL_PureOp<"rank"> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_IntScalar);
}

def IREEInterpHL_DimOp : IREEInterpHL_PureOp<"dim"> {
  // TODO(benvanik) add dim attr (I32Attr:$dim)
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_IntScalar);
}

def IREEInterpHL_ShapeOp : IREEInterpHL_PureOp<"shape"> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_1DIntMemRef);
}

def IREEInterpHL_LengthOp : IREEInterpHL_PureOp<"length"> {
  let arguments = (ins IREEHL_MemRef);
  let results = (outs IREEHL_IndexScalar);
}

def IREEInterpHL_SliceOp :
    IREEInterpHL_PureOp<"slice", [AllElementTypesMatch<["src", "result"]>,
                                  AllTypesMatch<["srcIndices", "lengths"]>]> {
  let arguments = (ins
      IREEHL_MemRef:$src,
      IREEHL_1DIndexMemRef:$srcIndices,
      IREEHL_1DIndexMemRef:$lengths
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_CopyOp : IREEInterpHL_Op<"copy", [
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

def IREEInterpHL_CloneOp :
    IREEInterpHL_PureOp<"clone", [SameOperandsAndResultType]> {
  let arguments = (ins IREEHL_MemRef:$src);
  let results = (outs IREEHL_MemRef);

  let hasFolder = 1;
}

// A pseudo op provided for convenience. This gets canonicalized to a series of
// copies.
def IREEInterpHL_ConcatOp : IREEInterpHL_PureOp<"concat"> {
  let arguments = (ins
      Variadic<IREEHL_MemRef>:$srcs,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_MemRef);

  let hasCanonicalizer = 1;
}

// TODO(benvanik): add split dim/size/etc. Maybe make multiple ops?
def IREEInterpHL_SplitOp :
    IREEInterpHL_PureOp<"split", [SameOperandsAndResultElementType]> {
  let arguments = (ins IREEHL_MemRef:$src);
  let results = (outs Variadic<IREEHL_MemRef>);
}

def IREEInterpHL_AssignOp :
    IREEInterpHL_PureOp<"assign", [SameOperandsAndResultType]> {
  let arguments = (ins IREEHL_MemRef:$src);
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_CondAssignOp :
    IREEInterpHL_PureOp<"cond_assign",
                        [AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins
      IREEHL_BoolScalar:$cond,
      IREEHL_MemRef:$lhs,
      IREEHL_MemRef:$rhs
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_ReshapeOp : IREEInterpHL_PureOp<"reshape"> {
  let arguments = (ins IREEHL_MemRef:$src, IREEHL_MemRef:$shape);
  let results = (outs IREEHL_MemRef);
}

def IREEInterpHL_SelectOp :
    IREEInterpHL_PureOp<"select", [AllTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins
      IREEHL_BoolMemRef:$cond,
      IREEHL_MemRef:$lhs,
      IREEHL_MemRef:$rhs
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_BroadcastOp :
    IREEInterpHL_PureOp<"broadcast",
                        [AllElementTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    IREE_ScalarMemRefOf<[AnyType]>:$operand,
    IREEHL_1DIntMemRef:$shape
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_PadOp :
    IREEInterpHL_PureOp<
        "pad", [AllElementTypesMatch<["src", "result", "padding_value"]>]> {
  let arguments = (ins
      IREEHL_MemRef:$src,
      IREEHL_AnyScalar:$padding_value,
      IREEHL_1DIndexMemRef:$edge_padding_low,
      IREEHL_1DIndexMemRef:$edge_padding_high,
      IREEHL_1DIndexMemRef:$interior_padding
  );

  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_TileOp :
    IREEInterpHL_PureOp<"tile", [AllElementTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    IREEHL_MemRef:$operand,
    IREEHL_1DIntMemRef:$shape
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_TransposeOp :
    IREEInterpHL_PureOp<"transpose", [
      AllElementTypesMatch<["operand", "result"]>,
      AllRanksMatch<["operand", "result"]>,
      AllElementCountsMatch<["operand", "result"]>
    ]> {
  let arguments = (ins
    IREEHL_MemRef:$operand,
    IREEHL_1DIntMemRef:$permutation
  );
  let results = (outs IREEHL_MemRef:$result);
}

def IREEInterpHL_ReverseOp :
    IREEInterpHL_PureOp<"reverse", [AllTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    IREEHL_MemRef:$operand,
    IREEHL_1DIntMemRef:$dims
  );
  let results = (outs IREEHL_MemRef:$result);
}

class IREEInterpHL_UnaryElementwiseOp<string mnemonic, Type type,
                                      list<OpTrait> traits = []> :
    IREEInterpHL_PureOp<mnemonic,
                        !listconcat(traits, [SameOperandsAndResultType])> {
  let arguments = (ins type);
  let results = (outs type);
}

class IREEInterpHL_UnaryElementwiseFloatOp<string mnemonic,
                                           list<OpTrait> traits = []> :
    IREEInterpHL_UnaryElementwiseOp<mnemonic, IREEHL_FloatMemRef, traits>;

class IREEInterpHL_UnaryElementwiseIntOp<string mnemonic,
                                           list<OpTrait> traits = []> :
    IREEInterpHL_UnaryElementwiseOp<mnemonic, IREEHL_IntMemRef, traits>;

class IREEInterpHL_BinaryElementwiseOp<string mnemonic, Type type,
                                       list<OpTrait> traits> :
    IREEInterpHL_PureOp<mnemonic,
                        !listconcat(traits, [SameOperandsAndResultType])> {
  let arguments = (ins type:$lhs, type:$rhs);
  let results = (outs type);
}

class IREEInterpHL_BinaryElementwiseFloatOp<string mnemonic,
                                           list<OpTrait> traits = []> :
    IREEInterpHL_BinaryElementwiseOp<mnemonic, IREEHL_FloatMemRef,
                                     traits>;

class IREEInterpHL_BinaryElementwiseIntOp<string mnemonic,
                                           list<OpTrait> traits = []> :
    IREEInterpHL_BinaryElementwiseOp<mnemonic, IREEHL_IntMemRef,
                                     traits>;

class IREEInterpHL_TernaryOp<string mnemonic,
                       Type type = IREEHL_MemRef,
                       list<OpTrait> traits = []> :
    IREEInterpHL_PureOp<mnemonic, traits> {
  let arguments = (ins type:$a, type:$b, type:$c);
  let results = (outs type);
}

// TODO(benvanik): add traits for broadcasting support.

def IREEInterpHL_NotOp : IREEInterpHL_UnaryElementwiseIntOp<"not">;
def IREEInterpHL_AndOp : IREEInterpHL_BinaryElementwiseIntOp<"and">;
def IREEInterpHL_OrOp : IREEInterpHL_BinaryElementwiseIntOp<"or">;
def IREEInterpHL_XorOp : IREEInterpHL_BinaryElementwiseIntOp<"xor">;
def IREEInterpHL_ShiftLeftOp : IREEInterpHL_BinaryElementwiseIntOp<"sll">;
def IREEInterpHL_ShiftRightLogicalOp : IREEInterpHL_BinaryElementwiseIntOp<"srl">;
def IREEInterpHL_ShiftRightArithmeticOp : IREEInterpHL_BinaryElementwiseIntOp<"sra">;

def IREEInterpHL_AddIOp : IREEInterpHL_BinaryElementwiseIntOp<"add_i">;
def IREEInterpHL_AddFOp : IREEInterpHL_BinaryElementwiseFloatOp<"add_f">;
def IREEInterpHL_SubIOp : IREEInterpHL_BinaryElementwiseIntOp<"sub_i">;
def IREEInterpHL_SubFOp : IREEInterpHL_BinaryElementwiseFloatOp<"sub_f">;
def IREEInterpHL_AbsIOp : IREEInterpHL_UnaryElementwiseIntOp<"abs_i">;
def IREEInterpHL_AbsFOp : IREEInterpHL_UnaryElementwiseFloatOp<"abs_f">;
def IREEInterpHL_MulIOp : IREEInterpHL_BinaryElementwiseIntOp<"mul_i">;
def IREEInterpHL_MulFOp : IREEInterpHL_BinaryElementwiseFloatOp<"mul_f">;
def IREEInterpHL_DivISOp : IREEInterpHL_BinaryElementwiseIntOp<"div_i_s">;
def IREEInterpHL_DivIUOp : IREEInterpHL_BinaryElementwiseIntOp<"div_i_u">;
def IREEInterpHL_DivFOp : IREEInterpHL_BinaryElementwiseFloatOp<"div_f">;
def IREEInterpHL_RemISOp : IREEInterpHL_BinaryElementwiseIntOp<"rem_i_s">;
def IREEInterpHL_RemIUOp : IREEInterpHL_BinaryElementwiseIntOp<"rem_i_u">;
def IREEInterpHL_RemFOp : IREEInterpHL_BinaryElementwiseFloatOp<"rem_f">;
def IREEInterpHL_MulAddIOp : IREEInterpHL_TernaryOp<"madd_i", IREEHL_IntMemRef>;
def IREEInterpHL_MulAddFOp : IREEInterpHL_TernaryOp<"madd_f", IREEHL_FloatMemRef>;
def IREEInterpHL_ExpFOp : IREEInterpHL_UnaryElementwiseFloatOp<"exp_f">;
def IREEInterpHL_LogFOp : IREEInterpHL_UnaryElementwiseFloatOp<"log_f">;
def IREEInterpHL_RsqrtFOp : IREEInterpHL_UnaryElementwiseFloatOp<"rsqrt_f">;
def IREEInterpHL_CosFOp : IREEInterpHL_UnaryElementwiseFloatOp<"cos_f">;
def IREEInterpHL_SinFOp : IREEInterpHL_UnaryElementwiseFloatOp<"sin_f">;
def IREEInterpHL_TanhFOp : IREEInterpHL_UnaryElementwiseFloatOp<"tanh_f">;
def IREEInterpHL_Atan2FOp : IREEInterpHL_UnaryElementwiseFloatOp<"atan2_f">;

def IREEInterpHL_MinISOp : IREEInterpHL_BinaryElementwiseIntOp<"min_i_s">;
def IREEInterpHL_MinIUOp : IREEInterpHL_BinaryElementwiseIntOp<"min_i_u">;
def IREEInterpHL_MinFOp : IREEInterpHL_BinaryElementwiseFloatOp<"min_f">;
def IREEInterpHL_MaxISOp : IREEInterpHL_BinaryElementwiseIntOp<"max_i_s">;
def IREEInterpHL_MaxIUOp : IREEInterpHL_BinaryElementwiseIntOp<"max_i_u">;
def IREEInterpHL_MaxFOp : IREEInterpHL_BinaryElementwiseFloatOp<"max_f">;
def IREEInterpHL_ClampFOp : IREEInterpHL_TernaryOp<"clamp_f", IREEHL_FloatMemRef>;
def IREEInterpHL_FloorFOp : IREEInterpHL_UnaryElementwiseFloatOp<"floor_f">;
def IREEInterpHL_CeilFOp : IREEInterpHL_UnaryElementwiseFloatOp<"ceil_f">;

class IREEInterpHL_ConversionOp<string mnemonic, Type inputType,
                                Type outputType> :
    IREEInterpHL_PureOp<mnemonic, [SameOperandsAndResultShape]> {
  let arguments = (ins inputType);
  let results = (outs outputType);
}

def IREEInterpHL_ConvertSSOp :
    IREEInterpHL_ConversionOp<"convert_s_s", IREEHL_IntMemRef,
                              IREEHL_IntMemRef>;
def IREEInterpHL_ConvertSUOp :
    IREEInterpHL_ConversionOp<"convert_s_u", IREEHL_IntMemRef,
                              IREEHL_IntMemRef>;
def IREEInterpHL_ConvertSFOp :
    IREEInterpHL_ConversionOp<"convert_s_f", IREEHL_IntMemRef,
                              IREEHL_FloatMemRef>;

def IREEInterpHL_ConvertUSOp :
    IREEInterpHL_ConversionOp<"convert_u_s", IREEHL_IntMemRef,
                              IREEHL_IntMemRef>;
def IREEInterpHL_ConvertUUOp :
    IREEInterpHL_ConversionOp<"convert_u_u", IREEHL_IntMemRef,
                              IREEHL_IntMemRef>;
def IREEInterpHL_ConvertUFOp :
    IREEInterpHL_ConversionOp<"convert_u_f", IREEHL_IntMemRef,
                              IREEHL_FloatMemRef>;

def IREEInterpHL_ConvertFSOp :
    IREEInterpHL_ConversionOp<"convert_f_s", IREEHL_FloatMemRef,
                              IREEHL_IntMemRef>;
def IREEInterpHL_ConvertFUOp :
    IREEInterpHL_ConversionOp<"convert_f_u", IREEHL_FloatMemRef,
                              IREEHL_IntMemRef>;
def IREEInterpHL_ConvertFFOp :
    IREEInterpHL_ConversionOp<"convert_f_f", IREEHL_FloatMemRef,
                              IREEHL_FloatMemRef>;

def IREEInterpHL_MatMulIOp :
    IREEInterpHL_PureOp<"matmul_i",
                        [AllElementTypesMatch<["lhs", "rhs", "result"]>]> {
  let arguments = (ins
      IREEHL_IntMemRef:$lhs,
      IREEHL_IntMemRef:$rhs,
      IREEHL_IntMemRef:$multiplier_mantissa,
      IREEHL_IntMemRef:$multiplier_exponent
  );
  let results = (outs IREEHL_IntMemRef:$result);
}
def IREEInterpHL_MatMulFOp :
    IREEInterpHL_PureOp<"matmul_f", [SameOperandsAndResultElementType]> {
  let arguments = (ins
      IREEHL_FloatMemRef:$lhs,
      IREEHL_FloatMemRef:$rhs
  );
  let results = (outs IREEHL_FloatMemRef);
}

def IREEInterpHL_ReduceSumIOp :
    IREEInterpHL_PureOp<"reduce_sum_i",
                        [AllElementTypesMatch<["src", "result", "init"]>]> {
  let arguments = (ins
      IREEHL_IntMemRef:$src,
      IREEHL_IntMemRef:$init,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_IntMemRef:$result);
}
def IREEInterpHL_ReduceSumFOp :
    IREEInterpHL_PureOp<"reduce_sum_f",
                        [AllElementTypesMatch<["src", "result", "init"]>]> {
  let arguments = (ins
      IREEHL_FloatMemRef:$src,
      IREEHL_FloatMemRef:$init,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_FloatMemRef:$result);
}
def IREEInterpHL_ReduceMinIOp :
    IREEInterpHL_PureOp<"reduce_min_i",
                        [AllElementTypesMatch<["src", "result", "init"]>]> {
  let arguments = (ins
      IREEHL_IntMemRef:$src,
      IREEHL_IntMemRef:$init,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_IntMemRef:$result);
}
def IREEInterpHL_ReduceMinFOp :
    IREEInterpHL_PureOp<"reduce_min_f",
                        [AllElementTypesMatch<["src", "result", "init"]>]> {
  let arguments = (ins
      IREEHL_FloatMemRef:$src,
      IREEHL_FloatMemRef:$init,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_FloatMemRef:$result);
}
def IREEInterpHL_ReduceMaxIOp :
    IREEInterpHL_PureOp<"reduce_max_i",
                        [AllElementTypesMatch<["src", "result", "init"]>]> {
  let arguments = (ins
      IREEHL_IntMemRef:$src,
      IREEHL_IntMemRef:$init,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_IntMemRef:$result);
}
def IREEInterpHL_ReduceMaxFOp :
    IREEInterpHL_PureOp<"reduce_max_f",
                        [AllElementTypesMatch<["src", "result", "init"]>]> {
  let arguments = (ins
      IREEHL_FloatMemRef:$src,
      IREEHL_FloatMemRef:$init,
      I32Attr:$dimension
  );
  let results = (outs IREEHL_FloatMemRef:$result);
}

def IREEInterpHL_TraceOp : IREEInterpHL_Op<"trace"> {
  let arguments = (ins Variadic<IREEHL_MemRef>:$srcs);
}

def IREEInterpHL_CondBreakOp : IREEInterpHL_Op<"cond_break"> {
  let arguments = (ins IREEHL_BoolScalar:$cond);
}

def IREEInterpHL_BreakOp : IREEInterpHL_Op<"break">;

#endif  // IREE_INTERPRETER_HL_OPS

// Common IREE op definitions shared by the interpreter and sequencer dialects.

#ifdef IREE_OP_BASE
#else
#define IREE_OP_BASE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif  // OP_BASE

//===----------------------------------------------------------------------===//
// Top-level IREE dialect
//===----------------------------------------------------------------------===//

def IREE_Dialect : Dialect {
  let name = "iree";
  let cppNamespace = "IREE";
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

class IREE_ScalarMemRefOf<list<Type> allowedTypes> :
    MemRefRankOf<allowedTypes, [0]>;

//===----------------------------------------------------------------------===//
// Op traits
//===----------------------------------------------------------------------===//

def IREE_IsModuleOnlyPred :
  CPred<"llvm::isa_and_nonnull<IREE::ModuleOp>($_op.getParentOp())">;
def IREE_ModuleOnly :
  PredOpTrait<"op can only be used in an 'iree.module' block",
              IREE_IsModuleOnlyPred>;

def IREE_IsMultiArchExecutableOnlyPred :
  CPred<"llvm::isa_and_nonnull<IREE::MultiArchExecutableOp>($_op.getParentOp())">;
def IREE_MultiArchExecutableOnly :
  PredOpTrait<"op can only be used in an 'iree.multi_arch_executable' block",
              IREE_IsMultiArchExecutableOnlyPred>;

def IREE_IsExecutableOnlyPred :
  CPred<"llvm::isa_and_nonnull<IREE::ExecutableOp>($_op.getParentOp())">;
def IREE_ExecutableOnly :
  PredOpTrait<"op can only be used in an 'iree.executable' block",
              IREE_IsExecutableOnlyPred>;

def IREE_IsExecutableTargetConfigOnlyPred :
  CPred<"llvm::isa_and_nonnull<IREE::ExecutableTargetConfigOp>($_op.getParentOp())">;
def IREE_ExecutableTargetConfigOnly :
  PredOpTrait<"op can only be used in an 'iree.target_config' block",
              IREE_IsExecutableTargetConfigOnlyPred>;

//===----------------------------------------------------------------------===//
// Structural types
//===----------------------------------------------------------------------===//

def IREE_IsExecutableType :
    CPred<"$_self.isa<::mlir::iree_compiler::ExecutableType>()">;

def IREE_Executable : Type<IREE_IsExecutableType, "an IREE executable">;

//===----------------------------------------------------------------------===//
// High-level types
//===----------------------------------------------------------------------===//

def IREEHL_Bool :
    AnyTypeOf<[I1, I8], "boolean-storing type (1 or 8 -bit integer)">;
def IREEHL_Element : AnyTypeOf<[AnyInteger, AnyFloat]>;

def IREEHL_MemRef : MemRefOf<[IREEHL_Element]>;
def IREEHL_BoolMemRef : MemRefOf<[IREEHL_Bool]>;
def IREEHL_IntMemRef : MemRefOf<[AnyInteger]>;
def IREEHL_FloatMemRef : MemRefOf<[AnyFloat]>;
def IREEHL_IndexMemRef : MemRefOf<[AnyInteger]>;

def IREEHL_AnyScalar : IREE_ScalarMemRefOf<[IREEHL_Element]>;
def IREEHL_BoolScalar : IREE_ScalarMemRefOf<[IREEHL_Bool]>;
def IREEHL_IntScalar : IREE_ScalarMemRefOf<[AnyInteger]>;
def IREEHL_FloatScalar : IREE_ScalarMemRefOf<[AnyFloat]>;
def IREEHL_IndexScalar : IREE_ScalarMemRefOf<[AnyInteger]>;
def IREEHL_I32Scalar : IREE_ScalarMemRefOf<[I32]>;

def IREEHL_1DIntMemRef : MemRefRankOf<[AnyInteger], [1]>;
def IREEHL_1DIndexMemRef : MemRefRankOf<[AnyInteger], [1]>;


//===----------------------------------------------------------------------===//
// Low-level types
//===----------------------------------------------------------------------===//

def IREELL_Bool : TypeAlias<I8, "boolean-storing type (8-bit integer)">;
def IREELL_Int: AnyTypeOf<[I8, I16, I32, I64], "8/16/32/64-bit integer">;
def IREELL_Float: AnyTypeOf<[F32, F64], "32/64-bit float">;
def IREELL_Index : AnyTypeOf<[I32, I64], "32/64-bit index integer">;
def IREELL_Element : AnyTypeOf<[IREELL_Int, IREELL_Float]>;

def IREELL_MemRef : MemRefOf<[IREELL_Element]>;
def IREELL_IntMemRef : MemRefOf<[IREELL_Int]>;
def IREELL_FloatMemRef : MemRefOf<[IREELL_Float]>;
def IREELL_BoolMemRef : MemRefOf<[IREELL_Bool]>;
def IREELL_IndexMemRef : MemRefOf<[IREELL_Index]>;
// For shape computation outputs, we want to consistently output I32 not I64
// TODO(b/138851470) Iron out story for index types
def IREELL_I32MemRef : MemRefOf<[I32]>;

def IREELL_ElementScalar : IREE_ScalarMemRefOf<[IREELL_Element]>;
def IREELL_IntScalar : IREE_ScalarMemRefOf<[IREELL_Int]>;
def IREELL_BoolScalar : IREE_ScalarMemRefOf<[IREELL_Bool]>;
def IREELL_FloatScalar : IREE_ScalarMemRefOf<[IREELL_Float]>;
def IREELL_IndexScalar : IREE_ScalarMemRefOf<[IREELL_Index]>;
// For shape computation outputs, we want to consistently output I32 not I64
// TODO(b/138851470) Iron out story for index types
def IREELL_I32Scalar : IREE_ScalarMemRefOf<[I32]>;

def IREELL_1DIntMemRef : MemRefRankOf<[IREELL_Int], [1]>;
def IREELL_1DIndexMemRef : MemRefRankOf<[IREELL_Index], [1]>;


//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

// Values are FourCC codes.
// See https://github.com/google/iree/tree/master/iree/iree/hal/executable_format.h for more info.
// https://ubershmekel.github.io/fourcc-to-text/ can be used to generate the
// integer values for inclusion here.
//
// LINT.IfChange(executable_format)
def IREE_EF_Unspecified : I32EnumAttrCase<"Unspecified", 538976288>;
def IREE_EF_MlirText : I32EnumAttrCase<"MlirText", 1296845138>;
def IREE_EF_IreeBytecode : I32EnumAttrCase<"IreeBytecode", 1230128453>;
def IREE_EF_SpirV : I32EnumAttrCase<"SpirV", 1397773893>;
def IREE_ExecutableFormatAttr :
    I32EnumAttr<"ExecutableFormat", "IREE Executable format", [
      IREE_EF_Unspecified,
      IREE_EF_MlirText,
      IREE_EF_IreeBytecode,
      IREE_EF_SpirV,
    ]> {
  let returnType = "uint32_t";
  let convertFromStorage = "static_cast<uint32_t>($_self.getInt())";
  let cppNamespace = "::mlir::iree_compiler::IREE";
}
// LINT.ThenChange(//iree/iree/hal/executable_format.h:fourcc)

// Use no padding and clamp the window to the valid area, possibly stopping
// early prior to having covered all data.
def IREE_PM_ClampWindowToFit : I32EnumAttrCase<"ClampWindowToFit", 0>;
// Use initial values for padding when windows cross dimension boundaries.
def IREE_PM_PadBorder : I32EnumAttrCase<"PadBorder", 1>;
// Describes the padding applied for a windowed operation like convolution,
// where a window is placed inside a base area.
def IREE_PaddingModeAttr :
    I32EnumAttr<"PaddingMode", "Padding mode", [
      IREE_PM_ClampWindowToFit,
      IREE_PM_PadBorder,
    ]> {
  let returnType = "uint32_t";
  let convertFromStorage = "static_cast<uint32_t>($_self.getInt())";
  let cppNamespace = "::mlir::iree_compiler::IREE";
}

#endif  // IREE_OP_BASE

add_mlir_library(IREEPyDMToIREEPasses
  ConversionPass.cpp
  LoweringPatterns.cpp
  TypeConverter.cpp

  DEPENDS
  IREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEInputDialect
  IREEPyDMDialect
  MLIRArithmetic
  MLIRControlFlow
  MLIRIR
  MLIRMath
  MLIRStandard
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEPyDMToIREEPasses)

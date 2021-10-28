add_mlir_library(IREEDialectsIREEPyDMToIREEPasses
  ConversionPass.cpp
  LoweringPatterns.cpp
  TypeConverter.cpp

  DEPENDS
  MLIRIREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMDialect
  IREEDialectsIREEDialect
  MLIRArithmetic
  MLIRIR
  MLIRStandard
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEDialectsIREEPyDMToIREEPasses)

add_mlir_library(IREEPyDMRTLPasses
  LinkageAnalysis.cpp
  LinkRTLPass.cpp
  LowerToRTLPass.cpp

  DEPENDS
  IREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEInputDialect
  IREEPyDMDialect
  MLIRIR
  MLIRParser
  MLIRFuncDialect
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEPyDMRTLPasses)

add_mlir_library(IREEDialectsIREEPyDMRTLPasses
  LinkageAnalysis.cpp
  LinkRTLPass.cpp
  LowerToRTLPass.cpp

  DEPENDS
  MLIRIREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMDialect
  IREEDialectsIREEDialect
  MLIRIR
  MLIRParser
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEDialectsIREEPyDMRTLPasses)

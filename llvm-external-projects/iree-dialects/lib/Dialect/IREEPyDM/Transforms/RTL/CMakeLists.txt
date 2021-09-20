add_mlir_library(IREEDialectsIREEPyDMRTLPasses
  LowerToRTLPass.cpp

  DEPENDS
  MLIRIREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMDialect
  IREEDialectsIREEDialect
  MLIRIR
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEDialectsIREEPyDMRTLPasses)

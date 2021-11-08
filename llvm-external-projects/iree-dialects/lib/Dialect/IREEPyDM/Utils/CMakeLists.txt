add_mlir_library(IREEDialectsIREEPyDMUtils
  TypeInference.cpp

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMDialect
  MLIRIR
  MLIRStandard
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEDialectsIREEPyDMUtils)

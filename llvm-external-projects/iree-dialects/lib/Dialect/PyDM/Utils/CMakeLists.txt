add_mlir_library(IREEPyDMUtils
  TypeInference.cpp

  LINK_LIBS PUBLIC
  IREEPyDMDialect
  MLIRIR
  MLIRFuncDialect
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEPyDMUtils)

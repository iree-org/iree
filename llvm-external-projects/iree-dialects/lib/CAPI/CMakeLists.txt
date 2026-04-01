add_mlir_public_c_api_library(IREEDialectsCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  IREELinalgTransformDialect
  IREELinalgTransformDialectPasses
  MLIRIR
  MLIRLinalgTransformOps
  MLIRTransformDialect
)

iree_dialects_target_includes(IREEDialectsCAPI)

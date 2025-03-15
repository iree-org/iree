add_mlir_public_c_api_library(IREEDialectsCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgTransformOps
  MLIRTransformDialect
  IREELinalgTransformDialect
  IREELinalgTransformDialectPasses
)

iree_dialects_target_includes(IREEDialectsCAPI)

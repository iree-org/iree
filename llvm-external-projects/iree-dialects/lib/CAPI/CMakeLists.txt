add_mlir_public_c_api_library(IREEDialectsCAPI
  Dialects.cpp
  Utils.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  IREEInputDialect
  IREELinalgExtDialect
  IREEPyDMDialect
  IREEPyDMPasses
)

iree_dialects_target_includes(IREEDialectsCAPI)

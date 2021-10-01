add_mlir_public_c_api_library(IREEDialectsCAPI
  Dialects.cpp
  Utils.cpp
  LINK_LIBS PUBLIC
  MLIRIR
  IREEDialectsIREEDialect
  IREEDialectsIREEPyDMDialect
  IREEDialectsIREEPyDMPasses
)

iree_dialects_target_includes(IREEDialectsCAPI)

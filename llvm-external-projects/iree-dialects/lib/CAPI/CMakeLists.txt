add_mlir_public_c_api_library(IREEDialectsCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  IREEDialectsIREEDialect
)

iree_dialects_target_includes(IREEDialectsCAPI)

add_mlir_public_c_api_library(IREEDialectsCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  IREEDialectsIREEDialect
)

# TODO: Fix upstream so there is a way to know what the actual compile target
# is (versus prefixing with "obj." which is conditional).
iree_dialects_target_includes(obj.IREEDialectsCAPI)

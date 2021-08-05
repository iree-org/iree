add_mlir_library(IREEDialectsIREEPublicDialect
  IREEPublicDialect.cpp
  IREEPublicOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  MLIRIREEPublicOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

# TODO: Fix upstream so there is a way to know what the actual compile target
# is (versus prefixing with "obj." which is conditional).
iree_dialects_target_includes(obj.IREEDialectsIREEPublicDialect)

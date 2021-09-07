add_mlir_library(IREEDialectsIREEPyDMDialect
  Dialect.cpp
  Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  MLIRIREEPyDMOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

iree_dialects_target_includes(IREEDialectsIREEPyDMDialect)

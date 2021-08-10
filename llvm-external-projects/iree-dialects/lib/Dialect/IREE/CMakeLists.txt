add_mlir_library(IREEDialectsIREEDialect
  IREEDialect.cpp
  IREEOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  MLIRIREEOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)

iree_dialects_target_includes(IREEDialectsIREEDialect)

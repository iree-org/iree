add_mlir_library(IREEPyDMDialect
  PyDMDialect.cpp
  PyDMOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREEPyDMIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRControlFlowInterfaces
  MLIRSideEffectInterfaces
)

iree_dialects_target_includes(IREEPyDMDialect)

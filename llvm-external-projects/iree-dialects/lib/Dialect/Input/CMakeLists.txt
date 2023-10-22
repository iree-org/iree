add_mlir_library(IREEInputDialect
  InputDialect.cpp
  InputOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREEInputIncGen
  IREEInputInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRTensorDialect
)

iree_dialects_target_includes(IREEInputDialect)

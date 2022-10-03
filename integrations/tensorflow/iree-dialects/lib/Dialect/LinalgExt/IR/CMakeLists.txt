add_mlir_library(IREELinalgExtDialect
  LinalgExtDialect.cpp
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREELinalgExtIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithUtils
  MLIRDialectUtils
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRTilingInterface
  MLIRViewLikeInterface
)

iree_dialects_target_includes(IREELinalgExtDialect)

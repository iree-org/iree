add_mlir_library(IREELinalgExtDialect
  LinalgExtDialect.cpp
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREELinalgExtIncGen

  LINK_LIBS PUBLIC
  IREELinalgExtUtils
  MLIRAffineDialect
  MLIRAffineUtils
  MLIRArithUtils
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalgDialect
  MLIRLinalgUtils
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRTensorDialect
  MLIRTensorUtils
  MLIRTilingInterface
  MLIRViewLikeInterface
)

iree_dialects_target_includes(IREELinalgExtDialect)

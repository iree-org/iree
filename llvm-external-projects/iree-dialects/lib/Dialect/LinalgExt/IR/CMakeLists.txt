add_mlir_library(IREELinalgExtDialect
  LinalgExtDialect.cpp
  LinalgExtInterfaces.cpp
  LinalgExtOps.cpp
  TiledOpInterface.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREELinalgExtIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithmeticUtils
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
  MLIRViewLikeInterface
)

iree_dialects_target_includes(IREELinalgExtDialect)

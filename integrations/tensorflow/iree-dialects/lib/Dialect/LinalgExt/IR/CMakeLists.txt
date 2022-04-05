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
  MLIRAffine
  MLIRArithmeticUtils
  MLIRDialectUtils
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLinalg
  MLIRMath
  MLIRMemRef
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRSCF
  MLIRFunc
  MLIRTensor
  MLIRViewLikeInterface
)

iree_dialects_target_includes(IREELinalgExtDialect)

add_mlir_library(IREELinalgExtTransformTestPasses
  TestLinalgExtTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREELinalgExtTransforms
  MLIRPass
  MLIRArithmetic
  MLIRMemRef
  MLIRSCF
  MLIRLinalg
  MLIRMath
  MLIRTensor
  )

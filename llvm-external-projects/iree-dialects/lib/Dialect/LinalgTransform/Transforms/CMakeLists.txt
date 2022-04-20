add_mlir_library(IREELinalgTransformDialectTransforms
  ExpertExpansion.cpp
  TrackingCSE.cpp
  TransformInterpreter.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREELinalgTransformDialect

  MLIRBufferization
  MLIRIR
  MLIRLinalg
  MLIRLLVMIR
  MLIRMath
  MLIRMathToLLVM
  MLIRMemRef
  MLIRMemRefToLLVM
  MLIRPass
  MLIRTensor
  MLIRTransforms
  MLIRVector
  MLIRVectorToLLVM
)

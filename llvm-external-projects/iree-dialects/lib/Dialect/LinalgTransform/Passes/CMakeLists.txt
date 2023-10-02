add_mlir_library(IREELinalgTransformDialectPasses
  TransformInterpreter.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREELinalgTransformDialect

  MLIRBufferizationDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRPass
  MLIRTensorDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToLLVM
)

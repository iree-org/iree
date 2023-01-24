add_mlir_library(IREELinalgTransformDialectPasses
  ExpertExpansion.cpp
  TransformInterpreter.cpp
  TransformInterpreterPassBase.cpp

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
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorToLLVM
)

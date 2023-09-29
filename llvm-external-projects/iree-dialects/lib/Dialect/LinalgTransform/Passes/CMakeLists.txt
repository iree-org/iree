add_mlir_library(IREELinalgTransformDialectPasses
  DropSchedule.cpp

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

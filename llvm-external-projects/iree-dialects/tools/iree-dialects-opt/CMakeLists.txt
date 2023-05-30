set(LIBS
  # Local dialects.
  IREEInputDialect
  IREELinalgExtDialect
  IREELinalgExtPasses
  IREELinalgExtTransformOps
  IREELinalgExtTransforms
  IREELinalgTransformDialect
  IREELinalgTransformDialectPasses
  IREELinalgTransformTestPasses
  IREETransformsTestPasses
  # Core dialects.
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRBufferizationTransformOps
  MLIRControlFlowDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRIndexToLLVM
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIROptLib
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRSCFDialect
  MLIRSCFTransformOps
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
  MLIRTransformPDLExtension
  MLIRVectorTransformOps
)

add_llvm_executable(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})

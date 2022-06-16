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
  IREEPyDMDialect
  IREEPyDMPasses
  # Core dialects.
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRAsyncDialect
  MLIRControlFlowDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRMemRefDialect
  MLIROptLib
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRSCFDialect
  MLIRSCFTransformOps
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
)

add_llvm_tool(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})

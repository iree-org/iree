set(LIBS
  # Local dialects.
  IREEInputDialect
  # Core dialects.
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRControlFlowDialect
  MLIRDialect
  MLIRFuncDialect
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
)

add_llvm_tool(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})

set(LIBS
  # Local dialects.
  IREEInputDialect
  IREELinalgExtDialect
  IREELinalgExtPasses
  IREELinalgExtTransforms
  IREELinalgTransformDialect
  IREELinalgTransformDialectTransforms
  IREELinalgTransformTestPasses
  IREETransformsTestPasses
  IREEPyDMDialect
  IREEPyDMPasses
  # Core dialects.
  MLIRArithmetic
  MLIRControlFlow
  MLIRDialect
  MLIRFunc
  MLIRLinalg
  MLIRMemRef
  MLIROptLib
  MLIRPDL
  MLIRPDLInterp
  MLIRSCF
  MLIRSCFTransforms
  MLIRTensor
  MLIRTransforms
)

add_llvm_tool(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})

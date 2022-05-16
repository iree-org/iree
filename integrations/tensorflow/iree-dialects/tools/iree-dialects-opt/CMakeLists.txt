set(LIBS
  # Local dialects.
  IREEInputDialect
  IREELinalgExtDialect
  IREELinalgExtOpInterfaceImpl
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
  MLIRAffine
  MLIRArithmetic
  MLIRAsync
  MLIRControlFlow
  MLIRDialect
  MLIRFunc
  MLIRLinalg
  MLIRLinalgTransformOps
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

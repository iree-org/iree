set(LIBS
  MLIRArithmetic
  MLIRControlFlow
  MLIRDialect
  MLIRLinalg
  MLIRMemRef
  MLIROptLib
  MLIRSCF
  MLIRSCFTransforms
  MLIRFunc
  MLIRTensor
  MLIRTransforms
  IREEInputDialect
  IREELinalgExtDialect
  IREELinalgExtPasses
  IREELinalgExtTransforms
  IREEPyDMDialect
  IREEPyDMPasses
)

add_llvm_tool(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})

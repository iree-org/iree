set(LIBS
  MLIRDialect
  MLIROptLib
  MLIRSCF
  MLIRSCFTransforms
  MLIRStandard
  MLIRTransforms
  IREEDialectsIREEDialect
  IREEDialectsIREEPyDMDialect
  IREEDialectsIREEPyDMPasses
)

add_llvm_tool(iree-dialects-opt
  iree-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(iree-dialects-opt PRIVATE ${LIBS})

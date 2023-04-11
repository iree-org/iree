
add_mlir_library(IREEDialectsTransforms
  ListenerCSE.cpp
  TransformMatchers.cpp

  LINK_LIBS PRIVATE
  # TODO: break dialect dependency by implementing the transformation separately
  # and registering it.
  MLIRAsyncDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect

  DEPENDS
  mlir-headers
  IREELinalgExtIncGen
  IREELinalgExtInterfacesIncGen
)

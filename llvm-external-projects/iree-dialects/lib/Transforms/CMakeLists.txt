
add_mlir_library(IREEDialectsTransforms
  TransformMatchers.cpp

  LINK_LIBS PRIVATE
  # TODO: break dialect dependency by implementing the transformation separately
  # and registering it.
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect

  DEPENDS
  mlir-headers
)

iree_dialects_target_includes(IREEDialectsTransforms)

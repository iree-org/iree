add_mlir_library(IREELinalgExtTransformOps
  LinalgExtTransformOps.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREEDialectsTransforms
  MLIRAsyncDialect
  MLIRRewrite
  MLIRTransformDialect
  IREELinalgExtDialect
  IREELinalgExtPasses
  IREELinalgExtTransforms

  MLIRPDLDialect
)

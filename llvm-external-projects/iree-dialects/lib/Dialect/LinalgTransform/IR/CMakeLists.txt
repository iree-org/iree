add_mlir_library(IREELinalgTransformDialect
  LinalgTransformOps.cpp
  PDL.cpp
  ScopedTransform.cpp
  TransformOpInterface.cpp
  TrackingListener.cpp
  TrackingRewriteDriver.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREEDialectsTransforms
  MLIRIR

  # Dialects
  IREELinalgExtDialect
  IREELinalgExtTransforms

  MLIRAsync
  MLIRLinalg
  MLIRPDL
  MLIRControlFlowInterfaces
  MLIRRewrite

  # Transforms
  MLIRAsyncTransforms
  MLIRLinalgTransforms
  MLIRAffineToStandard
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  
  # Conversions
  MLIRAsyncToLLVM
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRVectorToLLVM
  MLIRLinalgToLLVM
  MLIRSCFToControlFlow
)

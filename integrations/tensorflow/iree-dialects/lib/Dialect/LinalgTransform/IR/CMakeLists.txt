add_mlir_library(IREELinalgTransformDialect
  LinalgTransformOps.cpp
  ScopedTransform.cpp
  StructuredTransformOpsExt.cpp
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
  MLIRControlFlowInterfaces
  MLIRLinalg
  MLIRPDL
  MLIRRewrite
  MLIRTransformDialect

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

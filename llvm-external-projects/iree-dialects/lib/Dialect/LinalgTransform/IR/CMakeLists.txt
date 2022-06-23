add_mlir_library(IREELinalgTransformDialect
  LinalgTransformOps.cpp
  ScopedTransform.cpp
  StructuredTransformOpsExt.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREEDialectsTransforms
  MLIRIR

  # Dialects
  IREELinalgExtDialect
  IREELinalgExtTransforms

  MLIRAsyncDialect
  MLIRControlFlowInterfaces
  MLIRLinalgDialect
  MLIRPDLDialect
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

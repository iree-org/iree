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
  MLIRAffineToStandard
  MLIRAsyncTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRTransforms
  MLIRVectorToSCF
  
  # Conversions
  MLIRAsyncToLLVM
  MLIRIndexToLLVM
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRVectorToLLVM
  MLIRLinalgToLLVM
  MLIRSCFToControlFlow
)

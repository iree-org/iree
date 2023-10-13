add_mlir_library(IREELinalgTransformDialect
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
  MLIRTransformPDLExtension

  # Transforms
  MLIRAffineToStandard
  MLIRAsyncTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRReconcileUnrealizedCasts
  MLIRTensorTransformOps
  MLIRTransforms
  MLIRVectorToSCF

  # Conversions
  MLIRAsyncToLLVM
  MLIRIndexToLLVM
  MLIRMemRefToLLVM
  MLIRMathToLLVM
  MLIRVectorToLLVM
  MLIRVectorToLLVMPass
  MLIRSCFToControlFlow
)

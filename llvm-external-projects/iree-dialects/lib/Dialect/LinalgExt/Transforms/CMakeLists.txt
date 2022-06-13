add_mlir_library(IREELinalgExtTransforms
  ForeachThreadToAsync.cpp
  ForeachThreadToSequentialFor.cpp
  Fusion.cpp
  TilingExternalModels.cpp
  Tiling.cpp
  TilingToForeachThreadOp.cpp
  Utils.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  mlir-headers
  IREELinalgExtDialect

  LINK_LIBS PUBLIC
  IREELinalgExtDialect

  MLIRAffineToStandard
  MLIRAsync
  MLIRSCFToControlFlow
  MLIRLinalgToLLVM
  MLIRVectorToLLVM
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRIR
  MLIRMath
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRPass
  MLIRSCF
  MLIRTransforms
)

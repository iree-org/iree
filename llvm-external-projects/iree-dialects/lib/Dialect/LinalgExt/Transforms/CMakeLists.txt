add_mlir_library(IREELinalgExtTransforms
  InParallelToAsync.cpp
  InParallelToSequentialFor.cpp
  TilingExternalModels.cpp
  TileToSequentialFor.cpp
  TileToInParallel.cpp
  Tiling.cpp
  TilingToTileOp.cpp
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

add_mlir_library(IREELinalgExtOpInterfaceImpl
  LinalgExtBufferization.cpp

  PARTIAL_SOURCES_INTENDED
  LINK_LIBS PUBLIC
  IREELinalgExtDialect

  MLIRBufferization
  MLIRTensorTransforms
)

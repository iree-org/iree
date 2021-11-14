add_mlir_library(IREELinalgExtPasses
  ConvertToLoops.cpp
  Passes.cpp
  Tiling.cpp

  DEPENDS
  IREELinalgExtTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEInputDialect
  IREELinalgExtDialect
  MLIRAffine
  MLIRIR
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRMath
  MLIRMemRef
  MLIRPass
  MLIRSCF
  MLIRStandard
  MLIRSupport
  MLIRTensor
  MLIRTransforms
)

add_mlir_library(IREELinalgExtPasses
  ConvertToLoops.cpp
  PadContractionToBlockSize.cpp
  Passes.cpp
  Tiling.cpp

  DEPENDS
  IREELinalgExtPassesIncGen

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
  MLIRFunc
  MLIRSupport
  MLIRTensor
  MLIRTransforms
)

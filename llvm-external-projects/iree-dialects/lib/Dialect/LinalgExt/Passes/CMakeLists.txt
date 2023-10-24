add_mlir_library(IREELinalgExtPasses
  ConvertConv2DToWinograd.cpp
  ConvertToLoops.cpp
  DecomposeSoftmax.cpp
  MaterializeEncoding.cpp
  PadContractionToBlockSize.cpp
  Passes.cpp
  SplitReduction.cpp
  TileAndDecomposePass.cpp
  Tiling.cpp

  DEPENDS
  IREELinalgExtPassesIncGen

  LINK_LIBS PUBLIC
  IREEInputDialect
  IREELinalgExtDialect
  IREELinalgExtEncodingUtils
  IREELinalgExtUtils
  MLIRAffineDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

add_mlir_library(IREELinalgExtPasses
  ConvertToLoops.cpp
  PadContractionToBlockSize.cpp
  Passes.cpp
  SplitReduction.cpp
  Tiling.cpp

  DEPENDS
  IREELinalgExtPassesIncGen

  LINK_LIBS PUBLIC
  IREEInputDialect
  IREELinalgExtDialect
  MLIRAffineDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

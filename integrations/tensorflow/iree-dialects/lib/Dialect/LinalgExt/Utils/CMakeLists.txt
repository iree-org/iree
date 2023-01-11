add_mlir_library(IREELinalgExtUtils
  Utils.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  mlir-headers

  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRMemRefDialect
)

add_mlir_library(IREELinalgExtUtils
  Utils.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRMemRefDialect
)

add_mlir_library(IREELinalgExtEncodingUtils
  EncodingUtils.cpp

  LINK_LIBS PUBLIC
  IREELinalgExtDialect
  IREELinalgExtUtils
)

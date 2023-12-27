add_mlir_library(IREEVectorExtTestPasses
  TestIterators.cpp

  DEPENDS
  mlir-headers

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  IREEVectorExtDialect
  MLIRPass
  )


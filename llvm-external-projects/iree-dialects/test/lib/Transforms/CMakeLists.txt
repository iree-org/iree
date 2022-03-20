add_mlir_library(IREETransformsTestPasses
  TestListenerPasses.cpp

  DEPENDS
  mlir-headers

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  IREELinalgTransformDialect
  MLIRPass
  )

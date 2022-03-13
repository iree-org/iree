add_mlir_library(IREELinalgTransformTestPasses
  TestScopedTransform.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  IREELinalgTransformDialectTransforms
  MLIRPass
  )

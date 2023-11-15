add_mlir_library(IREEVectorExtDialect
  VectorExtDialect.cpp
  VectorExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREEVectorExtIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

iree_dialects_target_includes(IREEVectorExtDialect)

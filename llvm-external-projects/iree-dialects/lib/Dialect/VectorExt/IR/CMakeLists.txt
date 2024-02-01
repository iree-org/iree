add_mlir_library(IREEVectorExtDialect
  VectorExtAttrs.cpp
  VectorExtDialect.cpp
  VectorExtOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${IREE_DIALECTS_SOURCE_DIR}/include

  DEPENDS
  IREEVectorExtAttrsIncGen
  IREEVectorExtIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)

iree_dialects_target_includes(IREEVectorExtDialect)

add_mlir_library(IREEPyDMOptimizePasses
  FixateWeakNumeric.cpp
  LocalPropagateTypes.cpp
  VariablesToSSA.cpp

  DEPENDS
  IREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEPyDMDialect
  IREEPyDMUtils
  MLIRIR
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEPyDMOptimizePasses)

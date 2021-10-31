add_mlir_library(IREEDialectsIREEPyDMOptimizePasses
  FixateWeakNumeric.cpp
  LocalPropagateTypes.cpp
  VariablesToSSA.cpp

  DEPENDS
  MLIRIREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMDialect
  IREEDialectsIREEPyDMUtils
  MLIRIR
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEDialectsIREEPyDMOptimizePasses)

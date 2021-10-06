add_mlir_library(IREEDialectsIREEPyDMOptimizePasses
  FixateWeakNumeric.cpp

  DEPENDS
  MLIRIREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEDialectsIREEPyDMDialect
  MLIRIR
  MLIRTransformUtils
)

iree_dialects_target_includes(IREEDialectsIREEPyDMOptimizePasses)

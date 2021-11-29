add_subdirectory(Optimize)
add_subdirectory(RTL)
add_subdirectory(ToIREE)

add_mlir_library(IREEPyDMPasses
  Passes.cpp

  DEPENDS
  IREEPyDMTransformsPassesIncGen

  LINK_LIBS PUBLIC
  IREEPyDMOptimizePasses
  IREEPyDMRTLPasses
  IREEPyDMToIREEPasses
  MLIRTransforms
)

iree_dialects_target_includes(IREEPyDMPasses)

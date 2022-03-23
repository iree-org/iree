
add_mlir_library(IREEDialectsTransforms
  Listener.cpp
  ListenerCSE.cpp
  ListenerGreedyPatternRewriteDriver.cpp
  
  LINK_LIBS PRIVATE
  # TODO: break dialect dependency by implementing the transformation separately
  # and registering it.
  MLIRAsync
  MLIRLinalg
  MLIRLinalgTransforms

  DEPENDS
  mlir-headers
  IREELinalgExtIncGen
  IREELinalgExtInterfacesIncGen
)

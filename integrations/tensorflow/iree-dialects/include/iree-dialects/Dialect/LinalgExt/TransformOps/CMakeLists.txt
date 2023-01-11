function(_add_transform_dialect_extension)
  set(LLVM_TARGET_DEFINITIONS LinalgExtTransformOps.td)
  mlir_tablegen(LinalgExtTransformOps.h.inc -gen-op-decls)
  mlir_tablegen(LinalgExtTransformOps.cpp.inc -gen-op-defs)
  add_public_tablegen_target(IREELinalgExtTransformOpsIncGen)
  add_dependencies(mlir-headers IREELinalgExtTransformOpsIncGen)
endfunction()

_add_transform_dialect_extension()

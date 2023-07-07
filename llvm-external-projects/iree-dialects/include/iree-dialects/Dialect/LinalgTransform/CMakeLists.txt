function(_add_dialect)
  set(LLVM_TARGET_DEFINITIONS LinalgTransformOps.td)
  mlir_tablegen(LinalgTransformOps.h.inc -gen-op-decls)
  mlir_tablegen(LinalgTransformOps.cpp.inc -gen-op-defs)
  mlir_tablegen(LinalgTransformDialect.h.inc --gen-dialect-decls --dialect=iree_linalg_transform)
  mlir_tablegen(LinalgTransformDialect.cpp.inc --gen-dialect-defs --dialect=iree_linalg_transform)
  add_public_tablegen_target(IREELinalgTransformIncGen)
  add_dependencies(mlir-headers IREELinalgTransformIncGen)
endfunction()

function(_add_transform_dialect_extension)
  set(LLVM_TARGET_DEFINITIONS StructuredTransformOpsExt.td)
  mlir_tablegen(StructuredTransformOpsExt.h.inc -gen-op-decls)
  mlir_tablegen(StructuredTransformOpsExt.cpp.inc -gen-op-defs)
  add_public_tablegen_target(IREELinalgTransformExtIncGen)
  add_dependencies(mlir-headers IREELinalgTransformExtIncGen)
endfunction()

function(_add_transform_doc)
  set(LLVM_TARGET_DEFINITIONS LinalgTransformOps.td)
  mlir_tablegen(LinalgTransformOps.md -gen-dialect-doc)
  set(GEN_DOC_FILE ${IREE_DIALECTS_BINARY_DIR}/docs/Dialects/LinalgTransformOps.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/LinalgTransformOps.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LinalgTransformOps.md)
  add_custom_target(LinalgTransformOpsDocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(iree-dialects-doc LinalgTransformOpsDocGen)
endfunction()

function(_add_structured_transform_doc)
  set(LLVM_TARGET_DEFINITIONS StructuredTransformOpsExt.td)
  mlir_tablegen(StructuredTransformOpsExt.md -gen-dialect-doc)
  set(GEN_DOC_FILE ${IREE_DIALECTS_BINARY_DIR}/docs/Dialects/StructuredTransformOpsExt.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/StructuredTransformOpsExt.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/StructuredTransformOpsExt.md)
  add_custom_target(StructuredTransformOpsExtDocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(iree-dialects-doc StructuredTransformOpsExtDocGen)
endfunction()

_add_dialect()
_add_transform_dialect_extension()
_add_transform_doc()
_add_structured_transform_doc()

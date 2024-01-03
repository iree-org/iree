function(_add_interfaces)
  set(LLVM_TARGET_DEFINITIONS VectorExtInterfaces.td)
  mlir_tablegen(VectorExtAttrInterfaces.h.inc -gen-attr-interface-decls)
  mlir_tablegen(VectorExtAttrInterfaces.cpp.inc -gen-attr-interface-defs)
  add_public_tablegen_target(IREEVectorExtInterfacesIncGen)
  add_dependencies(IREEVectorExtIncGen IREEVectorExtInterfacesIncGen)
endfunction()

function(_add_dialect)
  set(LLVM_TARGET_DEFINITIONS VectorExtOps.td)
  mlir_tablegen(VectorExtAttrs.h.inc -gen-attrdef-decls)
  mlir_tablegen(VectorExtAttrs.cpp.inc -gen-attrdef-defs)
  mlir_tablegen(VectorExtEnums.h.inc -gen-enum-decls)
  mlir_tablegen(VectorExtEnums.cpp.inc -gen-enum-defs)
  mlir_tablegen(VectorExtOps.h.inc -gen-op-decls)
  mlir_tablegen(VectorExtOps.cpp.inc -gen-op-defs)
  mlir_tablegen(VectorExtTypes.h.inc -gen-typedef-decls)
  mlir_tablegen(VectorExtTypes.cpp.inc -gen-typedef-defs)
  mlir_tablegen(VectorExtDialect.h.inc --gen-dialect-decls --dialect=iree_vector_ext)
  mlir_tablegen(VectorExtDialect.cpp.inc --gen-dialect-defs --dialect=iree_vector_ext)
  add_public_tablegen_target(IREEVectorExtIncGen)
  add_dependencies(mlir-headers IREEVectorExtIncGen)
endfunction()

function(_add_doc)
  set(LLVM_TARGET_DEFINITIONS VectorExtOps.td)
  set(_FLAGS
    "--strip-prefix=::mlir::iree_compiler::IREE::"
  )
  mlir_tablegen(VectorExtOps.md -gen-dialect-doc ${_FLAGS})
  set(GEN_DOC_FILE ${IREE_DIALECTS_BINARY_DIR}/docs/Dialects/VectorExtOps.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/VectorExtOps.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/VectorExtOps.md)
  add_custom_target(VectorExtOpsDocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(iree-dialects-doc VectorExtOpsDocGen)
endfunction()

_add_dialect()
_add_interfaces()
_add_doc()

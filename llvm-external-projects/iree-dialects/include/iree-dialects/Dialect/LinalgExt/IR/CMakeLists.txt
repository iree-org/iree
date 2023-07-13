function(_add_interfaces)
  set(LLVM_TARGET_DEFINITIONS LinalgExtInterfaces.td)
  mlir_tablegen(LinalgExtOpInterfaces.h.inc -gen-op-interface-decls)
  mlir_tablegen(LinalgExtOpInterfaces.cpp.inc -gen-op-interface-defs)
  mlir_tablegen(LinalgExtTypeInterfaces.h.inc -gen-type-interface-decls)
  mlir_tablegen(LinalgExtTypeInterfaces.cpp.inc -gen-type-interface-defs)
  add_public_tablegen_target(IREELinalgExtInterfacesIncGen)
  add_dependencies(IREELinalgExtIncGen IREELinalgExtInterfacesIncGen)
endfunction()

function(_add_dialect)
  set(LLVM_TARGET_DEFINITIONS LinalgExtOps.td)
  mlir_tablegen(LinalgExtAttrs.h.inc -gen-attrdef-decls)
  mlir_tablegen(LinalgExtAttrs.cpp.inc -gen-attrdef-defs)
  mlir_tablegen(LinalgExtEnums.h.inc -gen-enum-decls)
  mlir_tablegen(LinalgExtEnums.cpp.inc -gen-enum-defs)
  mlir_tablegen(LinalgExtOps.h.inc -gen-op-decls)
  mlir_tablegen(LinalgExtOps.cpp.inc -gen-op-defs)
  mlir_tablegen(LinalgExtTypes.h.inc -gen-typedef-decls)
  mlir_tablegen(LinalgExtTypes.cpp.inc -gen-typedef-defs)
  mlir_tablegen(LinalgExtDialect.h.inc --gen-dialect-decls --dialect=iree_linalg_ext)
  mlir_tablegen(LinalgExtDialect.cpp.inc --gen-dialect-defs --dialect=iree_linalg_ext)
  add_public_tablegen_target(IREELinalgExtIncGen)
  add_dependencies(mlir-headers IREELinalgExtIncGen)
endfunction()

function(_add_doc)
  set(LLVM_TARGET_DEFINITIONS LinalgExtOps.td)
  set(_FLAGS
    "--strip-prefix=::mlir::iree_compiler::IREE::"
  )
  mlir_tablegen(LinalgExtOps.md -gen-dialect-doc ${_FLAGS})
  set(GEN_DOC_FILE ${IREE_DIALECTS_BINARY_DIR}/docs/Dialects/LinalgExtOps.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/LinalgExtOps.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/LinalgExtOps.md)
  add_custom_target(LinalgExtOpsDocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(iree-dialects-doc LinalgExtOpsDocGen)
endfunction()

_add_dialect()
_add_interfaces()
_add_doc()

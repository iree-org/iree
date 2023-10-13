function(_add_interfaces)
  set(LLVM_TARGET_DEFINITIONS InputInterfaces.td)
  mlir_tablegen(InputOpInterfaces.h.inc --gen-op-interface-decls)
  mlir_tablegen(InputOpInterfaces.cpp.inc --gen-op-interface-defs)
  add_public_tablegen_target(IREEInputInterfacesIncGen)
  add_dependencies(mlir-headers IREEInputInterfacesIncGen)
endfunction()

function(_add_dialect)
  set(LLVM_TARGET_DEFINITIONS InputOps.td)
  mlir_tablegen(InputOps.h.inc -gen-op-decls)
  mlir_tablegen(InputOps.cpp.inc -gen-op-defs)
  mlir_tablegen(InputTypes.h.inc -gen-typedef-decls)
  mlir_tablegen(InputTypes.cpp.inc -gen-typedef-defs)
  mlir_tablegen(InputAttrs.h.inc -gen-attrdef-decls)
  mlir_tablegen(InputAttrs.cpp.inc -gen-attrdef-defs)
  mlir_tablegen(InputEnums.h.inc -gen-enum-decls)
  mlir_tablegen(InputEnums.cpp.inc -gen-enum-defs)
  mlir_tablegen(InputDialect.h.inc --gen-dialect-decls --dialect=iree_input)
  mlir_tablegen(InputDialect.cpp.inc --gen-dialect-defs --dialect=iree_input)
  add_public_tablegen_target(IREEInputIncGen)
  add_dependencies(mlir-headers IREEInputIncGen)
endfunction()

function(_add_doc)
  set(LLVM_TARGET_DEFINITIONS InputOps.td)
  set(_FLAGS
    "--strip-prefix=::mlir::iree_compiler::IREE::"
  )
  mlir_tablegen(InputOps.md -gen-dialect-doc ${_FLAGS})
  set(GEN_DOC_FILE ${IREE_DIALECTS_BINARY_DIR}/docs/Dialects/InputOps.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/InputOps.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/InputOps.md)
  add_custom_target(InputOpsDocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(iree-dialects-doc InputOpsDocGen)
endfunction()

_add_interfaces()
_add_dialect()
_add_doc()

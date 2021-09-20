# Generates runtime library resources.

file(GLOB_RECURSE _python_sources
  ${IREE_DIALECTS_SOURCE_DIR}/python/iree/compiler/dialects/iree_pydm/*.py
)

add_custom_target(IREEPyDMRTLGen
  DEPENDS
    IREEDialectsPythonModules
  SOURCES ${_python_sources}
)

function(rtl_builder)
  cmake_parse_arguments(ARG
    ""
    "MLIR_FILE"
    "RTL_MODULES"
    ${ARGN})

  add_custom_command(
    TARGET IREEPyDMRTLGen PRE_BUILD
    BYPRODUCTS "${ARG_MLIR_FILE}"
    COMMAND
      "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${IREE_DIALECTS_BINARY_DIR}/python_packages/iree_dialects"
      "${Python3_EXECUTABLE}" -m iree.compiler.dialects.iree_pydm.rtl.rtl_builder
      --output "${ARG_MLIR_FILE}"
      ${ARG_RTL_MODULES}
    COMMENT "Generating PyDM RTL: ${ARG_MLIR_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  )
endfunction()

rtl_builder(
  MLIR_FILE PyDMRTLBase.mlir
  RTL_MODULES
    .booleans
    .numerics
)

llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(IREE_DIALECTS_TEST_DEPENDS
        FileCheck count not
        iree-dialects-opt
        )

if(MLIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND IREE_DIALECTS_TEST_DEPENDS
    IREEDialectsPythonModules
    IREEPyDMRTLGen
  )
endif()

add_lit_testsuite(check-iree-dialects "Running the iree-dialects regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${IREE_DIALECTS_TEST_DEPENDS}
        )
set_target_properties(check-iree-dialects PROPERTIES FOLDER "Tests")

add_lit_testsuites(IREE_DIALECTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${IREE_DIALECTS_TEST_DEPENDS})

add_subdirectory(lib)
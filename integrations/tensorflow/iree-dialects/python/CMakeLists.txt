include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under npcomp.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=iree.compiler.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(IREEDialectsPythonSources)
declare_mlir_python_sources(IREEDialectsPythonExtensions)

declare_mlir_python_sources(IREEDialectsPythonSources.Dialects
  ADD_TO_PARENT IREEDialectsPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEDialectsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/IreeInputBinding.td
  SOURCES dialects/iree_input.py
  DIALECT_NAME iree_input
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEDialectsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/IreeLinalgExtBinding.td
  SOURCES dialects/iree_linalg_ext.py
  DIALECT_NAME iree_linalg_ext
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT IREEDialectsPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  TD_FILE dialects/IreePyDmBinding.td
  SOURCES
    dialects/_iree_pydm_ops_ext.py
    dialects/iree_pydm/__init__.py
  SOURCES_GLOB
    dialects/iree_pydm/importer/*.py
    dialects/iree_pydm/rtl/*.py
  DIALECT_NAME iree_pydm
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(IREEDialectsPythonExtensions.Main
  MODULE_NAME _ireeDialects
  ADD_TO_PARENT IREEDialectsPythonExtensions
  SOURCES
    IREEDialectsModule.cpp
  EMBED_CAPI_LINK_LIBS
    IREEDialectsCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Passes
  IREEDialectsPythonSources
  IREEDialectsPythonExtensions
)

add_mlir_python_common_capi_library(IREEDialectsAggregateCAPI
  INSTALL_COMPONENT IREEDialectsPythonModules
  INSTALL_DESTINATION python_packages/iree_dialects/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_DIALECTS_BINARY_DIR}/python_packages/iree_dialects/iree/compiler/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(IREEDialectsPythonModules
  ROOT_PREFIX "${IREE_DIALECTS_BINARY_DIR}/python_packages/iree_dialects/iree/compiler"
  INSTALL_PREFIX "python_packages/iree_dialects/iree/compiler"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    IREEDialectsAggregateCAPI
  )

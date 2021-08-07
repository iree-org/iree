include(AddMLIRPython)

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
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
  TD_FILE dialects/IreePublicBinding.td
  SOURCES dialects/iree_public.py
  DIALECT_NAME iree_public
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(IREEDialectsPythonExtensions.Main
  MODULE_NAME _ireeDialects
  ADD_TO_PARENT IREEDialectsPythonExtensions
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/IREEDialectsModule.cpp
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
  IREEDialectsPythonSources
  IREEDialectsPythonExtensions
)

add_mlir_python_common_capi_library(IREEDialectsAggregateCAPI
  INSTALL_COMPONENT IREEDialectsPythonModules
  INSTALL_DESTINATION python_packages/iree_dialects/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_DIALECTS_BINARY_DIR}/python_packages/iree_dialects/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(IREEDialectsPythonModules
  ROOT_PREFIX "${IREE_DIALECTS_BINARY_DIR}/python_packages/iree_dialects/mlir"
  INSTALL_PREFIX "python_packages/iree_dialects/mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    IREEDialectsAggregateCAPI
  )

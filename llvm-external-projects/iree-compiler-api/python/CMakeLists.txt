include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under npcomp.
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=iree.compiler.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(IREECompilerAPIPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  SOURCES
    api/driver.py
    version.py
)
declare_mlir_python_sources(IREECompilerAPIPythonExtensions)

declare_mlir_python_sources(IREECompilerAPIPythonTools
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/iree/compiler"
  SOURCES
    __init__.py
    tf.py
    tflite.py
    xla.py
  SOURCES_GLOB
    tools/*.py
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(IREECompilerAPIPythonExtensions.CompilerDriver
  MODULE_NAME _ireeCompilerDriver
  ADD_TO_PARENT IREECompilerAPIPythonExtensions
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CompilerModule.cpp
  EMBED_CAPI_LINK_LIBS
    IREECompilerAPICompilerCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
################################################################################

set(_source_components
  # Local sources.
  IREECompilerAPIPythonSources
  IREECompilerAPIPythonExtensions
  IREECompilerAPIPythonTools

  # TODO: Core is now implicitly building/registering all dialects, increasing
  # build burden by ~5x. Make it stop.
  MLIRPythonSources.Core

  # Passes should also be disaggregated.
  MLIRPythonSources.Passes

  # Core dialects (constrained to IREE input dialects).
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.shape
  MLIRPythonSources.Dialects.std
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.tosa
  MLIRPythonSources.Dialects.vector

  # mhlo project.
  MLIRHLOPythonSources
  MLIRHLOPythonExtensions

  # iree-dialects project.
  IREEDialectsPythonSources
  IREEDialectsPythonExtensions
)

add_mlir_python_common_capi_library(IREECompilerAggregateCAPI
  INSTALL_COMPONENT IREECompilerPythonModules
  INSTALL_DESTINATION python_package/iree/compiler/_mlir_libs
  OUTPUT_DIRECTORY "${IREE_COMPILER_API_BINARY_DIR}/python_package/iree/compiler/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(IREECompilerPythonModules
  ROOT_PREFIX "${IREE_COMPILER_API_BINARY_DIR}/python_package/iree/compiler"
  INSTALL_PREFIX "python_package/iree/compiler"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    IREECompilerAggregateCAPI
  )


################################################################################
# Tools linked against the shared CAPI library
################################################################################

# Build the ireec tool into _mlir_libs.
add_executable(
  IREECompilerIREECTool
    IREECTool.c
)
target_link_libraries(IREECompilerIREECTool IREECompilerAggregateCAPI)
set_target_properties(IREECompilerIREECTool
  PROPERTIES
    OUTPUT_NAME "ireec"
    RUNTIME_OUTPUT_DIRECTORY "${IREE_COMPILER_API_BINARY_DIR}/python_package/iree/compiler/_mlir_libs"
)
mlir_python_setup_extension_rpath(IREECompilerIREECTool)
add_dependencies(IREECompilerPythonModules IREECompilerIREECTool)

# Install tools.
install(TARGETS IREECompilerIREECTool
  DESTINATION "python_package/iree/compiler/_mlir_libs"
)

# LLD may not have been enabled at the global level (typically because we
# are not building for a CPU backend). If not, don't add API support for it.
set(_OPTIONAL_SOURCES)
set(_OPTIONAL_LINK_LIBS)
if(TARGET lld)
  list(APPEND _OPTIONAL_SOURCES Lld.cpp)
  list(APPEND _OPTIONAL_LINK_LIBS
    lldCommon
    lldCOFF
    lldELF
    lldMachO
    lldMinGW
    lldWasm
  )
endif()

add_mlir_public_c_api_library(IREECompilerAPICompilerCAPI
  Compiler.cpp
  Tools.cpp
  ${_OPTIONAL_SOURCES}
  PARTIAL_SOURCES_INTENDED
  # TODO: If installing, complains about IREEVM not being in any export set.
  DISABLE_INSTALL
  LINK_COMPONENTS
    Support
  LINK_LIBS PUBLIC
    MLIRIR
    iree::compiler::ConstEval
    iree::compiler::InputConversion::MHLO::MHLO
    iree::compiler::InputConversion::TOSA::TOSA
    iree::compiler::Dialect::VM::IR::IR
    iree::compiler::Dialect::VM::Target::Bytecode::Bytecode
    iree::compiler::Pipelines

    # Passes and dialects.
    iree::tools::init_llvmir_translations
    iree::tools::init_passes_and_dialects

    # All HAL Targets.
    iree::tools::init_targets

    # Tools.
    iree::tools::iree_translate_lib

    ${_OPTIONAL_LINK_LIBS}
)

# TODO: Fix upstream so there is a way to know what the actual compile target
# is (versus prefixing with "obj." which is conditional).
iree_compiler_target_includes(obj.IREECompilerAPICompilerCAPI)

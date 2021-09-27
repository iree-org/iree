add_mlir_public_c_api_library(IREECompilerAPICompilerCAPI
  Compiler.cpp
  Tools.cpp
  # TODO: If installing, complains about IREEVM not being in any export set.
  DISABLE_INSTALL
  LINK_COMPONENTS
    Support
  LINK_LIBS PUBLIC
    MLIRIR
    iree::compiler::Dialect::VM::IR::IR
    iree::compiler::Dialect::VM::Target::Bytecode::Bytecode
    iree::compiler::Translation::IREEVM

    # All HAL Targets.
    iree::tools::init_targets

    # Tools.
    iree::tools::iree_translate_lib
)

# TODO: Fix upstream so there is a way to know what the actual compile target
# is (versus prefixing with "obj." which is conditional).
iree_compiler_target_includes(obj.IREECompilerAPICompilerCAPI)

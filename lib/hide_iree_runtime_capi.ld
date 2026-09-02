# Version script for libIREECompiler (Linux/ELF).
#
# The compiler lib bundles the IREE runtime, so it also carries the runtime's
# `iree_*` C-API under its real names, the rename reaching only mangled C++.
# Compile-then-run flows load both libraries at once, and a runtime plugin's
# undefined iree_hal_* may then bind to the compiler's copy, leaving its device
# without the loaders the real runtime registered.
#
# So hide every snake_case `iree_*` from the export table. Nothing else resolves
# here by that spelling: the embedding C-API is camelCase, the MLIR C-API is
# mlir*, and the renamed internals are _ZN6IREE18*. exported_symbols_test.py
# keeps it that way.
{
  global: *;
  local: iree_*;
};

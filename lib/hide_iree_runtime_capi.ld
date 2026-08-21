# Version script for libIREECompiler (Linux/ELF).
#
# The compiler lib statically bundles the IREE runtime (local target backends,
# VM emission, etc.), so it also contains the runtime's `extern "C"` C-API
# (iree_hal_*, iree_vm_*, iree_status_*, iree_allocator_*, ...). The symbol
# rename only rewrites mangled llvm/mlir C++ symbols, so these C symbols keep
# their real names. When the compiler lib is loaded in the same process as the
# real IREE runtime (iree.runtime / _runtime.so) -- compile-then-run flows do
# exactly this -- the two copies collide. A runtime plugin's undefined
# iree_hal_* can then bind to the compiler's copy instead of the runtime's,
# leaving its device without the executable loaders the runtime registered.
#
# Hide every snake_case `iree_*` symbol from the dynamic export table so the
# compiler lib can never shadow the co-resident runtime. Nothing embedders or
# plugins resolve here is snake_case `iree_*`: the compiler embedding C-API is
# camelCase (ireeCompiler*, ireeCodegen*), the MLIR C-API is mlir*, and the
# renamed C++ internals plugins link are _ZN6IREE18*/_Z6IREE18* -- none match
# `iree_*`, so all stay exported. lib/exported_symbols_test.py enforces this
# split, so a convention violation fails CI instead of silently shadowing.
{
  global: *;
  local: iree_*;
};

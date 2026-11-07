// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-vmvx-split-reduction{reassociate-fp-reductions=true reduction-chunk-size=1}))' --verify-diagnostics %s

// expected-error@+1 {{reduction-chunk-size must be at least 2}}
func.func @invalid_chunk_size() {
  return
}

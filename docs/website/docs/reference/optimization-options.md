---
icon: octicons/rocket-16
---

# Optimization options

This page documents various supported flags for optimizing IREE programs. Each
is presented with its English name, flag to enable/disable, and default state.

These flags can be passed to the:

* `iree-compile` command line tool
* `extra_args=["--flag"]` argument to `iree.compiler.tools` Python wrappers
* In-process Python compiler API
  `iree.compiler.transforms.iree-compile.CompilerOptions("--flag", "--flag2")`
  constructor
* `ireeCompilerOptionsSetFlags()` compiler C API function

## Optimization level

As in other compilers like clang and gcc, IREE provides a high level optimization
level flag (`iree-opt-level`) that enables different sets of underlying options.

`iree-opt-level` specifies the optimization level for the entire compilation
flow. Lower optimization levels prioritize debuggability and stability, while
higher levels focus on maximizing performance. By default, `iree-opt-level` is
set to `O0` (minimal or no optimizations).

This flag takes the following values:

| Optimization Level | Pros | Cons |
|-------------------|------|------|
| **O0** (Default, Minimal Optimizations) | - Fastest compilation time.  <br> - Generated code is easier to debug. <br> - Keeps assertions enabled | - Poor runtime performance. <br> - Higher runtime memory usage. <br> - Larger code size due to lack of optimization. |
| **O1** (Basic Optimizations) | - Enables optimizations, allowing for better runtime performance. <br> - Optimizations are compatable with all backends.  | - Only applies conservative optimizations. <br> - Reduced debuggability. |
| **O2** (Optimizations without full backend support) | - Even more aggressive optimizations. <br> - Strikes a balance between optimization level and compatibility. | - Some optimizations may not be supported by all backends. <br> - Reduced debuggability. |
| **O3** (Aggressive Optimization) | - Highest runtime performance.  <br> - Enables advanced and aggressive transformations.  <br> - Exploits backend-specific optimizations for optimal efficiency. | - Longer compile times.  <br> - Some optimizations may be unstable. <br> - Reduced debuggability. |

Although `iree-opt-level` sets the default for each subflag, they can be
explicitly set on or off independently.

For example:

```sh
// Apply the default optimizations of `O2` but don't remove assertions.
iree-compile --iree-opt-level=O2 --iree-strip-assertions=false

// Minimize optimizations, but still preform aggressive fusion.
iree-compile --iree-opt-level=O0 --iree-dispatch-creation-enable-aggressive-fusion=true
```

## High level program optimizations

### Constant evaluation (`--iree-opt-const-eval` (on))

Performs compile-time evaluation of any global initializers which produce
the initial values for global constants, storing the global directly in the
program as constant data. This extracts such constant program fragments and
recursively compiles them, using the runtime to evaluate the results.

Note that this only has any effect on computations in module initializer
functions, not free-standing operations in the program which may produce
constant-derived results. See `--iree-opt-const-expr-hoisting` for options to
optimize these.

### Constant expression hoisting (`--iree-opt-const-expr-hoisting` (off))

Identifies all trees of constant expressions in the program and uses a
heuristic to determine which would be profitable to hoist into global
initializers for evaluation at module load. Together with
`--iree-opt-const-eval`, this will convert eligible trees of expressions to
purely static data embedded in the module.

The heuristic is currently relatively primitive, using static information to
disable hoisting of leaf operations which are metadata only (i.e.
broadcasts, etc) or are expected to fold away as part of operator fusion.
Notably, the current heuristic is likely to pessimize module size in the case of
complicated programs with trees of constant, large tensors.

### Numeric precision reduction (`--iree-opt-numeric-precision-reduction` (off))

Analyzes program constant data and program flow to identify math operations
which can be safely evaluated with reduced precision (currently with a minimum
of 8bit integers but being extended to infer any bit depth) and inserts
appropriate casts. In conjunction with *Constant Expression Hoisting*,
*Constant Evaluation* and other automatic optimizations, this can produce
programs where large amounts (up to the whole) have had their numeric operations
and constant data rewritten to lower precision types.

This feature is actively evolving and will be the subject of dedicated
documentation when ready.

### Strip Debug Assertions (`--iree-opt-strip-assertions` (off))

Strips all `std.assert` ops in the input program after useful information for
optimization analysis has been extracted. Assertions provide useful user-visible
error messages but can prevent critical optimizations. Assertions are not,
however, a substitution for control flow and frontends that want to check errors
in optimized release builds should do so via actual code - similar to when one
would `if (foo) return false;` vs. `assert(foo);` in a normal program.

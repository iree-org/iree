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

## High level program optimizations

### Constant evaluation (`--iree-opt-const-eval` (off))

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

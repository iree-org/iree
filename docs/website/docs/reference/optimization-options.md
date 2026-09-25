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

!!! note

    Not all flags that control performance are nested under `iree-opt-level`.
    See [High level program optimizations](#high-level-program-optimizations)
    below for subflags not covered by optimization flags.

This flag takes the following values:

| Optimization Level | Pros | Cons |
|-------------------|------|------|
| **O0** (Default, Minimal Optimizations) | <ul style="list-style-type:none;"><li>✔️ Fastest compilation time.</li><li>✔️ Generated code is easier to debug.</li><li>✔️ Keeps assertions enabled</li></ul> | <ul style="list-style-type:none;"><li>❌ Poor runtime performance.</li><li>❌ Higher runtime memory usage.</li><li>❌ Larger code size due to lack of optimization.</li></ul> |
| **O1** (Basic Optimizations) | <ul style="list-style-type:none;"><li>✔️ Enables optimizations, allowing for better runtime performance.</li><li>✔️ Optimizations are compatible with all backends.</li></ul> | <ul style="list-style-type:none;"><li>➖ Only applies conservative optimizations.</li><li>❌ Reduced debuggability.</li></ul> |
| **O2** (Optimizations without full backend support) | <ul style="list-style-type:none;"><li>✔️ Even more aggressive optimizations.</li><li>✔️ Strikes a balance between optimization level and compatibility.</li></ul> | <ul style="list-style-type:none;"><li>➖ Some optimizations may not be supported by all backends.</li><li>❌ Reduced debuggability.</li></ul> |
| **O3** (Aggressive Optimization) | <ul style="list-style-type:none;"><li>✔️ Highest runtime performance.</li><li>✔️ Enables advanced and aggressive transformations.</li><li>✔️ Exploits backend-specific optimizations for optimal efficiency.</li></ul> | <ul style="list-style-type:none;"><li>➖ Longer compile times.</li><li>❌ Some optimizations may be unstable.</li><li>❌ Reduced debuggability.</li></ul> |

Although `iree-opt-level` sets the default for each subflag, they can be
explicitly set on or off independently.

For example:

```bash
# Apply the default optimizations of `O2` but don't remove assertions.
iree-compile --iree-opt-level=O2 --iree-strip-assertions=false

# Minimize optimizations, but still preform aggressive fusion.
iree-compile --iree-opt-level=O0 --iree-dispatch-creation-enable-aggressive-fusion=true
```

### Pipeline-level control

In addition to `iree-opt-level`, IREE provides optimization controls at the
pipeline level. These flags allow fine-grained tuning of specific compilation
stages while still respecting the topmost optimization level unless explicitly
overridden.

#### Dispatch Creation (`iree-dispatch-creation-opt-level`)

- `iree-dispatch-creation-enable-aggressive-fusion` (enabled at `O2`)

    Enables more aggressive fusion opportunities not yet supported by all backends

#### Global Optimization (`iree-global-optimization-opt-level`)

- `iree-opt-strip-assertions` (enabled at `O1`)

    Strips all `std.assert` ops in the input program after useful information for
    optimization analysis has been extracted. Assertions provide useful
    user-visible error messages but can prevent critical optimizations.
    Assertions are not, however, a substitution for control flow and frontends
    that want to check errors in optimized release builds should do so via
    actual code - similar to when one would `if (foo) return false;` vs.
    `assert(foo);` in a normal program.

- `iree-opt-outer-dim-concat` (enabled at `O1`)

    Transpose concat operations to occur along the outermost dimension. The
    resulting concat will now be contiguous and the inserted transposes can
    possibly be fused with surrounding ops.

- `iree-opt-aggressively-propagate-transposes` (enabled at `O3`)

    Enables more transpose propagation by allowing transposes to be propagated
    to `linalg` named ops even when the resulting op will be a `linalg.generic`.

## High level program optimizations

### Data-Tiling (`--iree-opt-data-tiling` (off))

Modifies data layout of operands for certain operations, such as matrix
multiplication, that prefer specifics layouts. These layout preferences depend
on the operations and the target hardware.

The layout changes can be propagated as far as possible across the workload, so
the entire workload can use the updated layouts, as opposed to having to perform
layout transformations at runtime. This may involve fusions or
constant-evaluation that can amortize or remove layout-transformation overheads.
See the [blog post](https://iree.dev/community/blog/2025-08-25-data-tiling-walkthrough/)
to learn more about details.

Data-tiling is an optional optimization technique, and it is not supported by
all backends. Any backend does not develop data-tiling may result in worse
performance if it is enabled. The targets that support data-tiling are:

* CPU (x86)
* CPU (AArch64)
* CPU (RISC-V)
* GPU (ROCm)
* VMVX

There are three paths in data-tiling:

* `--iree-dispatch-creation-data-tiling` (off): Starts data-tiling from dispatch
  creation phase. This is the current default path, if the flag is on. This is
  still under development and some features are missing. The main benefit from
  the path is that it enables more fusion opportunities and supports
  multi-devices.
* `--iree-global-opt-data-tiling` (off): Starts data-tiling from global
  optimization phase. The path will be deprecated in the future, because it
  breaks the core IREE concept in terms of retargetability and modularity. Any
  features built on top of the path may be deprecated.
* `--iree-opt-experimental-early-data-tiling` (off): Experimental. Assigns,
  propagates and materializes data-tiling layouts in the global optimization
  phase, so that the layouts propagate through the surrounding tensor graph and
  data-tiled ops such as `linalg.mmt4d` still fuse with their elementwise
  epilogues and result unpacks in dispatch creation. Unlike the global
  optimization path, it normalizes contraction dimensions and propagates layouts
  through elementwise ops before materializing them.

    This makes global optimization target aware on purpose, so the path is
    limited to modules that it can commit to a single target:

    * The module has a single executable target that supports materializing
      layouts on the host: currently `llvm-cpu` on x86_64 or AArch64, with the
      CPU encoding resolver and without inner-tiled layouts.
    * The encoding strategy is the default one
      (`--iree-dispatch-creation-set-encoding-strategy=generic`).

    Other modules fall back to the dispatch creation path. Because of that, the
    op families and the encoding strategy come from the dispatch creation flags
    (`--iree-dispatch-creation-experimental-set-data-tiling-ops` and
    `--iree-dispatch-creation-set-encoding-strategy`). A module whose layouts
    were materialized records its target in the
    `iree.encoding.materialized_layout_target` attribute, even if nothing was
    data-tiled. Dispatch creation then does not assign encodings again, and
    rejects compiling the module for any other target, e.g. when compilation is
    resumed from an intermediate phase.
    [#25031](https://github.com/iree-org/iree/issues/25031) tracks moving the
    layout decision into its own component that decides per affinity, which
    would lift these limits.

Note that `--iree-opt-data-tiling` will override the dispatch creation and
global optimization flags if it is explicitly set, because it has higher
priority. You have to disable the flag to use a specific path.
`--iree-opt-experimental-early-data-tiling` overrides all of them, so
conflicting data-tiling flags are not an error when it is set.

| Feature             | GlobalOptimization path | DispatchCreation path | Early path (experimental) |
|---------------------|-------------------------|-----------------------|---------------------------|
| ConstExpr Hoisting  | ✔️                      | ✔️                    | ✔️                        |
| Constant Evaluation | ✔️                      | ❌                    | ✔️                        |
| Matmul Fusion       | ❌                      | ✔️                    | ✔️                        |
| Multi-device        | ❌                      | ✔️                    | ❌                        |

Note: Constant Evaluation is not yet implemented in the dispatch creation path.
It is a planned feature.

!!! question - "Something missing or broken?"

    Don't see the target or the feature here that you think should be? Reach out
    to [our data-tiling Discord channel](https://discord.com/channels/689900678990135345/1254843174111678555);
    we welcome contributions on
    [our GitHub page](https://github.com/iree-org/iree)!

### Constant evaluation (`--iree-opt-const-eval` (on))

Performs compile-time evaluation of any global initializers which produce
the initial values for global constants, storing the global directly in the
program as constant data. This extracts such constant program fragments and
recursively compiles them, using the runtime to evaluate the results.

Note that this only has any effect on computations in module initializer
functions, not free-standing operations in the program which may produce
constant-derived results. See `--iree-opt-const-expr-hoisting` for options to
optimize these.

### Constant expression hoisting (`--iree-opt-const-expr-hoisting` (on))

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

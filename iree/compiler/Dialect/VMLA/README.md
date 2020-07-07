# VMLA (Virtual Machine-based Linear Algebra)

This dialect is designed to closely model XLA HLO ops in a way that is easy to
map to execution on the IREE VM. The changes involve using byte buffers instead
of tensors, propagating shape information and converting shape math to simple
integer arithmetic, and legalizing types to supported values (such as 1bit bools
to 8bit integers of 0 or 1).

## Adding an Op

As with other VM modules, VMLA ops are declared in
[vmla.imports.mlir](/iree/compiler/Dialect/VMLA/vmla.imports.mlir). These
declarations are what enable the compiler and runtime side to talk to each
other. It's helpful to start here to think about the information you need to
communicate to the runtime prior to writing conversions. As a general rule, try
to avoid communicating anything not strictly required for a correct
implementation; instead, perform more work on the compiler side if it allows
simpler ops to be implemented at runtime. For example, if there's an attribute
on the op that selects between two different implementations, instead of
plumbing that attribute through to runtime and switching there, one should
implement the conversion to lower it into two ops. This makes it easier to
reduce binary sizes, get accurate profiles at runtime, etc.

TLDR:

1.  Add a `vm.import` to
    [vmla.imports.mlir](/iree/compiler/Dialect/VMLA/vmla.imports.mlir).
2.  Add an MLIR op def to
    [VMLAOps.td](/iree/compiler/Dialect/VMLA/IR/VMLAOps.td).
3.  Add a conversion from the source dialect like
    [HLOToVMLA](/iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA/).
4.  Add a conversion to the `vm.import` in
    [VMLAToVM](/iree/compiler/Dialect/VMLA/Conversion/VMLAToVM/).
5.  Add the runtime C++ kernel thunk to
    [vmla_module.cc](/iree/hal/vmla/vmla_module.cc).
6.  Declare the kernel in [op_kernels.h](/iree/hal/vmla/op_kernels.h) and add a
    reference implementation in
    [op_kernels_generic.h](/iree/hal/vmla/op_kernels_generic.h).

### Declaring the Op

See the file comments in
[vmla.imports.mlir](/iree/compiler/Dialect/VMLA/vmla.imports.mlir) for style and
naming conventions. Note that **the suffix naming convention is load-bearing and
must be followed**.

Add a new `vm.import` corresponding to the op you want to add. Try to group it
with existing related ops in the file.

If the op does not need to know type information then always prefer to use `xN`
as the op suffix (like `copy.x8`, which copies 8-bit elements, as copies do not
care that the bits they are copying are ints or floats).

Almost all ops use output argument buffers (such as `%dst`). Only ops that
return references to in-place contents of existing input buffers should return
values (such as `buffer.view`).

If shape information is required (and it's a good idea to make sure it
absolutely is) then add shapes following the buffer they are related to, for
example: `vm.import @transpose.x8(%src : !vm.ref<!vmla.buffer>, %src_shape : i32
...)`

### Adding the Op Tablegen Description

Once the op is declared you can add the tablegen op def in
[VMLAOps.td](/iree/compiler/Dialect/VMLA/IR/VMLAOps.td). Match the order and
grouping in this file with the `vmla.imports.mlir` file to make moving between
the two easier.

The automated conversion helper uses names and order to match the op defs with
the `vm.import` declarations. Make sure the names of all argument values and
attributes match those in the declaration.

Many ops can be expressed with `VMLA_UnaryOp`/`VMLA_BinaryOp`/etc classes such
as `VMLA_AddOp`. These will automatically get their `lhs`/`rhs`/`dst` and fan
out to the given type group. For example, use `VMLA_AnyTypeAttr` will allow both
integers and floats of various bit depths, while `VMLA_FloatTypeAttr` will only
allow floating-point values. These should match to which suffixes you defined in
the import; for example if you only have `foo.f32` declared to indicate that the
op only operates on floating-point values then use `VMLA_FloatTypeAttr`).

For ops that don't fit the unary/binary/etc form you can use the
`VMLA_ElementTypeOp` class to get at least get the automated type suffix
conversion. These expect an argument of `VMLA_*TypeAttr:$element_type` to store
the appropriate type from the result value and it will be populated
automatically.

For ops that require shapes you must add the `VMLA_IncludeShapes` trait to tell
the automated conversion helper to insert shape information. Again, really try
to avoid passing shape information if possible (use element counts/etc that you
can derive from buffer sizes at runtime, if needed).

Finally, some ops like `VMLA_MatMulOp` may use multiple types and may need to
provide their own type extraction and suffix creation logic. For these add the
`VMLA_OpInterface` trait and define an `extractTypeAttributes` function.

### Converting to the VMLA Op

There are two conversion required: one from the source dialect to your new VMLA
op and one from the VMLA op to the VM import call.

See [HLOToVMLA](/iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA/) for examples
of the former. Most ops can use the `VMLAOpConversion` helper to automatically
convert between ops so long as they match in values and attributes (for example,
`mhlo.add` can be trivially converted to `vmla.add`). Examples of more complex
ops that may require additional IR to be emitted or attributes to be mapped can
be seen in there as well.

You can add tests for your conversion as needed under `test/` in the appropriate
dialect-specific conversion folder.

### Converting to the VM Import

If your new op is defined well then the conversion from VMLA to VM should be
straightforward. Many ops can use `VMLA_*_IMPORT_OP` macros to perform the
conversion automatically. See
[VMLAToVM](/iree/compiler/Dialect/VMLA/Conversion/VMLAToVM/) for examples.

*   Ops that have no suffix (like `vmla.buffer.fill`) use `VMLA_IMPORT_OP`.
*   Ops that only require the bit-depth (like `vmla.copy.x8`) use
    `VMLA_SIZED_IMPORT_OP`.
*   Ops that require type information (like `vmla.cmp.f32`) use
    `VMLA_TYPED_IMPORT_OP`.

Custom conversions can be performed as well but try to avoid that.

You can add tests for your conversion under the `test/` path and are encouraged
to do so particularly if not using the `VMLA_*_IMPORT_OP` macros.

### Add the Runtime Kernel

[vmla_module.cc](/iree/hal/vmla/vmla_module.cc) contains the runtime companion
of the `vmla.imports.mlir` file mapping from the VM calls to C++. Again add your
function in here in the same place as you did in the other files. Follow the
example of other functions in the file for how to declare arguments, how to add
the `IREE_TRACE_SCOPE` line, etc.

There are some helpers such as `IREE_VMLA_BINARY_OP` that match the equivalents
in the tablegen file such that if your op can usually be just a single line.

The thunks in this file just call one of the kernels defined in the
[op_kernels.h](/iree/hal/vmla/op_kernels.h) file. These kernels are designed to
be standalone from the VM code and take effectively just pointers and lists of
values. The job of the `vmla_module.cc` thunk is to unwrap the VM arguments and
pass them to these functions.

Declare your new kernel in the header without its implementation. If your kernel
needs to keep state at runtime you can follow what `MatMul` does with the
`RuntimeState` struct, however it is strongly discouraged and almost never
required, so avoid if possible. One way to avoid it is to make your op take any
scratch memory it may require as an argument and generate the IR during
conversion. This ensures that we can optimize things on the compiler-side
instead of forcing the runtime to deal with things.

Finally, implement the kernel in
[op_kernels_generic.h](/iree/hal/vmla/op_kernels_generic.h). Try to keep it
simple and readable. These are reference kernels and don't need to be fast,
however all of our tests use them and as such they shouldn't be so slow as to
prevent tests from running in a reasonable time. Use your judgement or be
willing to have someone file a bug telling you to make them faster if they are
terribly slow :)

Tests for the kernels can be added to
[op_kernels_test.cc](/iree/hal/vmla/op_kernels_test.cc). The thunks in
`vmla_module.cc` are best tested via end-to-end tests using `iree-run-mlir` as
what you really want to ensure is that the compiler is emitting calls that match
the runtime side and the only way to do this is to actually compile and run.

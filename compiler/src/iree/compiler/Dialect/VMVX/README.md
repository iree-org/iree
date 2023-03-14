# VMVX (Virtual Machine-based Vector eXtensions)

This dialect is designed as a virtual ISA (Instruction Set Architecture)
extension to the IREE VM that exposes variable-length vector operations.
The intent is that ops from the `std` and `vector` dialects can be mapped
to these operations instead of lowering them to scalar loops that would
otherwise be too expensive to execute in IREE bytecode.

The operations added here are modeled as close to a machine ISA as reasonable,
meaning that there are no shapes, element types are encoded as part of the
operations, and memory access is tightly restricted.

## Adding an Op

As with other VM modules, VMVX ops are declared in
[vmvx.imports.mlir](vmvx.imports.mlir).
These declarations are what enable the compiler and runtime side to talk to each
other. It's helpful to start here to think about the information you need to
communicate to the runtime prior to writing conversions. As a general rule try
to avoid communicating anything not strictly required for a correct
implementation; instead, perform more work on the compiler side if it allows
simpler ops to be implemented at runtime. For example, if there's an attribute
on the op that selects between two different implementations, instead of
plumbing that attribute through to runtime and switching there, one should
implement the conversion to lower it into two ops. This makes it easier to
reduce binary sizes, get accurate profiles at runtime, etc.

TLDR:

1.  Add an MLIR op def to
    [VMVXOps.td](IR/VMVXOps.td).
2.  Add a conversion from the source dialect like
    [StandardToVMVX](Conversion/StandardToVMVX/).
3.  Add a `vm.import` to
    [vmvx.imports.mlir](vmvx.imports.mlir).
4.  Add a conversion to the `vm.import` in
    [VMVXToVM](Conversion/VMVXToVM/).
5.  Add a definition matching the `vm.import` to the runtime module
    [exports.inl](/runtime/src/iree/modules/vmvx/exports.inl).
6.  Add the runtime method implementing the op to
    [vmvx_module.c](/runtime/src/iree/modules/vmvx/module.c).

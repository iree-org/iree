// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_VM_OPS
#define IREE_DIALECT_VM_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilInterfaces.td"
include "iree/compiler/Dialect/Util/IR/UtilTypes.td"
include "iree/compiler/Dialect/VM/IR/VMBase.td"

// Trial ops do not have any memory effects but may invoke Undefine Behavior on
// some inputs. This makes them conditionally speculatable/hoistable.
class VM_TrivialOp<string mnemonic, list<Trait> traits = []> :
      VM_Op<mnemonic, !listconcat(traits, [NoMemoryEffect])>;

// Pure ops do not have any memory effects, do not invoke Undefined Behavior,
// and are always safe to speculate/hoist.
//
// TODO(https://github.com/openxla/iree/issues/11179): More VM ops should be
// made pure.
class VM_PureOp<string mnemonic, list<Trait> traits = []> :
      VM_TrivialOp<mnemonic, !listconcat(traits, [AlwaysSpeculatable])>;

//===----------------------------------------------------------------------===//
// Structural ops
//===----------------------------------------------------------------------===//

def VM_ModuleOp : VM_Op<"module", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"IREE::VM::ModuleTerminatorOp">,
    NativeOpTrait<"SymbolTable">,
    Symbol,
  ]> {
  let summary = [{module containing VM functions and variables}];
  let description = [{
    Top-level container for VM functions.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    OptionalAttr<VM_OrdinalCountsAttr>:$ordinal_counts,
    OptionalAttr<I32Attr>:$version
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name
    attr-dict-with-keyword
    regions
  }];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name)>,
  ];

  let extraClassDeclaration = [{
    Region &getBodyRegion() { return getOperation()->getRegion(0); }
    Block& getBlock() { return this->getOperation()->getRegion(0).front(); }

    struct Dependency {
      StringRef name;
      uint32_t minimumVersion = 0;
      bool isOptional = true;
    };
    // Returns an alphabetically-ordered list of dependencies based on vm.import
    // ops and metadata.
    SmallVector<Dependency> getDependencies();
  }];

  let hasVerifier = 1;
}

def VM_ModuleTerminatorOp : VM_Op<"module_terminator", [
    HasParent<"IREE::VM::ModuleOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the module op}];

  let assemblyFormat = "attr-dict";
}

// TODO(benvanik): DCE vm.func when unused and not exported.
// TODO(benvanik): attributes for function analysis: pure, etc (for imports).
def VM_FuncOp : VM_Op<"func", [
    IsolatedFromAbove,
    HasParent<"IREE::VM::ModuleOp">,
    FunctionOpInterface,
    CallableOpInterface,
    Symbol,
  ]> {
  let summary = [{function defined with VM control flow ops}];
  let description = [{
    Represents a function containing VM ops and those of compatible dialects.
    All flow control is performed by VM ops.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<VM_Ordinal>:$ordinal,
    OptionalAttr<UnitAttr>:$noinline,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>,
  ];

  let extraClassDeclaration = [{
    /// Add an entry block to an empty function and set up the block arguments
    /// to match the signature of the function.
    Block *addEntryBlock();

    Block *addBlock() {
      assert(!empty() && "function should at least have an entry block");
      push_back(new Block());
      return &back();
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return &getBody(); }
    ArrayRef<Type> getCallableResults() {
      assert(!isExternal() && "invalid callable");
      return getFunctionType().getResults();
    }

    /// Adds or overrides a reflection attribute.
    void setReflectionAttr(StringRef name, Attribute value);
  }];
}

def VM_ExportOp : VM_Op<"export", [
    // TODO(#3968): working around iree.module.export hackability woes.
    // HasParent<"IREE::VM::ModuleOp">,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ]> {
  let summary = [{exports a function from the module}];
  let description = [{
    Specifies an exported function with an externally-visible alias. Multiple
    exports can reference the same internal functions.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$function_ref,
    StrAttr:$export_name,
    OptionalAttr<VM_Ordinal>:$ordinal
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "FuncOp":$functionRef,
      CArg<"StringRef", [{""}]>:$exportName,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "FlatSymbolRefAttr":$functionRef,
      CArg<"StringRef", [{""}]>:$exportName,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];
}

def VM_ImportOp : VM_Op<"import", [
    Symbol,
    FunctionOpInterface,
    CallableOpInterface,
  ]> {
  let summary = [{imports a function from an external module}];
  let description = [{
    Specifies a function that should be imported from either the runtime or
    an external VM module.

    Required imports can be declared with a minimum version of the module that
    contains the import. The maximum declared minimum version of all required
    imports from the module will become the required minimum version at runtime.

    Optional imports not present at runtime will be invalid to call and whether
    they were resolved can be queried with `vm.import.resolved`.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<VM_Ordinal>:$ordinal,
                       OptionalAttr<UnitAttr>:$is_optional,
                       OptionalAttr<I32Attr>:$minimum_version
  );

  // Required because FunctionOpInterface and CallableOpInterface disagree with
  // each other about the requirement for regions. We don't use this region.
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
      CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>,
  ];

  let extraClassDeclaration = [{
    /// Returns true if the import must be invoked via vm.call.variadic.
    bool isVariadic() {
      for (int i = 0; i < getArgumentTypes().size(); ++i) {
        if (isFuncArgumentVariadic(i)) return true;
      }
      return false;
    }

    /// Returns the name of the given function argument.
    StringRef getFuncArgumentName(int i) {
      return getArgAttrOfType<StringAttr>(i, "vm.name").getValue();
    }

    /// Returns true if the given function argument is a variadic/packed value.
    bool isFuncArgumentVariadic(int i) {
      return !!getArgAttrOfType<UnitAttr>(i, "vm.variadic");
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return nullptr; }
    ArrayRef<Type> getCallableResults() {
      return getFunctionType().getResults();
    }
  }];
}

def VM_InitializerOp : VM_Op<"initializer", [
    IsolatedFromAbove,
    HasParent<"IREE::VM::ModuleOp">,
    FunctionOpInterface,
    CallableOpInterface,
    Util_InitializerOpInterface,
  ]> {
  let summary = [{global initialization function}];
  let description = [{
    A function that is called in definition order upon module initialization.
    Must not load any globals that are defined or initialized after it in the
    module.
  }];

  let arguments = (ins
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
  ];

  let extraClassDeclaration = [{
    /// Add an entry block to an empty function and set up the block arguments
    /// to match the signature of the function.
    Block *addEntryBlock();
    Block *addBlock();

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return {}; }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return {}; }

    LogicalResult verifyType() { return success(); }

    Region *getCallableRegion() { return &getBody(); }
    ArrayRef<Type> getCallableResults() { return {}; }

    /// Make symbol optional as this op has no symbol.
    bool isOptionalSymbol() { return true; }
  }];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

class VM_PrimitiveGlobalOp<string mnemonic, Attr attr_type, list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      IsolatedFromAbove,
      HasParent<"IREE::VM::ModuleOp">,
      Symbol,
      Util_GlobalOpInterface,
    ])> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<attr_type>:$initial_value,
    OptionalAttr<VM_Ordinal>:$ordinal
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    (`mutable` $is_mutable^)?
    $sym_name
    attr-dict
    custom<TypeOrAttr>($type, $initial_value)
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name, "bool":$isMutable, "Type":$type,
      "Optional<TypedAttr>":$initialValue,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    ),
    [{
      $_state.addAttribute(SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
      if (isMutable) {
        $_state.addAttribute("is_mutable", $_builder.getUnitAttr());
      }
      if (initialValue.has_value() &&
                 (initialValue.value().isa<IntegerAttr>() ||
                  initialValue.value().isa<FloatAttr>())) {
        $_state.addAttribute("initial_value", initialValue.value());
      }
      $_state.addAttribute("type", TypeAttr::get(type));
      $_state.attributes.append(attrs.begin(), attrs.end());
    }]>,
    OpBuilder<(ins
      "StringRef":$name, "bool":$isMutable, "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    ),
    [{
      build($_builder, $_state, name, isMutable, type, std::nullopt, attrs);
    }]>,
  ];
}

def VM_GlobalI32Op : VM_PrimitiveGlobalOp<"global.i32",
                                          VM_ConstantIntegerValueAttr<I32>> {
  let summary = [{32-bit integer global declaration}];
  let description = [{
    Defines a global value that is treated as a scalar literal at runtime.
    Initialized to zero unless an initial value is specified.
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalI64Op : VM_PrimitiveGlobalOp<"global.i64",
                                          VM_ConstantIntegerValueAttr<I64>> {
  let summary = [{64-bit integer global declaration}];
  let description = [{
    Defines a global value that is treated as a scalar literal at runtime.
    Initialized to zero unless an initial value is specified.
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalF32Op : VM_PrimitiveGlobalOp<"global.f32",
                                          VM_ConstantFloatValueAttr<F32>,
                                          [VM_ExtF32]> {
  let summary = [{32-bit floating-point global declaration}];
  let description = [{
    Defines a global value that is treated as a scalar literal at runtime.
    Initialized to zero unless an initial value is specified.
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalF64Op : VM_PrimitiveGlobalOp<"global.f64",
                                          VM_ConstantFloatValueAttr<F64>,
                                          [VM_ExtF64]> {
  let summary = [{64-bit floating-point global declaration}];
  let description = [{
    Defines a global value that is treated as a scalar literal at runtime.
    Initialized to zero unless an initial value is specified.
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalRefOp :
    VM_Op<"global.ref", [
      IsolatedFromAbove,
      HasParent<"IREE::VM::ModuleOp">,
      Symbol,
      DeclareOpInterfaceMethods<Util_GlobalOpInterface>,
    ]> {
  let summary = [{ref<T> global declaration}];
  let description = [{
    Defines a global value that is a ref of a specific type. The global will
    retain the ref object for the lifetime of the context or until the value is
    replaced with a store or reset.
    Initialized to null unless an initial value is specified.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<VM_Ordinal>:$ordinal
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    (`mutable` $is_mutable^)?
    $sym_name
    attr-dict
    `:` $type
  }];

  let builders = [
    OpBuilder<(ins
      "StringRef":$name, "bool":$isMutable, "Type":$type,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    ),
    [{
      $_state.addAttribute(SymbolTable::getSymbolAttrName(),
                           $_builder.getStringAttr(name));
      if (isMutable) {
        $_state.addAttribute("is_mutable", $_builder.getUnitAttr());
      }
      $_state.addAttribute("type", TypeAttr::get(type));
      $_state.attributes.append(attrs.begin(), attrs.end());
    }]>,
  ];

  let extraClassDeclaration = [{
    // No initial values supported for ref globals (they start null).
    Attribute getGlobalInitialValue() { return {}; }
    void setGlobalInitialValue(Attribute) {}
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalAddressOp : VM_PureOp<"global.address", [
      SymbolUserOpInterface,
      Util_GlobalAddressOpInterface,
    ]> {
  let summary = [{returns an address reference to a global}];
  let description = [{
    Returns an indirect address reference to the given global. During export the
    address will be converted to the natural format of the global table (for
    example, ordinals for refs and byte offsets for primitive types).
  }];

  let arguments = (ins
    VM_GlobalRefAttr:$global
  );
  let results = (outs
    AnyTypeOf<[VM_Ptr, Util_AnyPtrType]>:$result
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable) {
      return IREE::Util::detail::verifyGlobalAddressOp(*this, symbolTable);
    }
  }];
}

class VM_GlobalLoadOp<Type type, string mnemonic, list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      SymbolUserOpInterface,
      Util_GlobalLoadOpInterface,
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      // HACK: works around the lack of symbol side effects in C++.
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    ])> {
  let arguments = (ins
    VM_GlobalRefAttr:$global
  );
  let results = (outs
    type:$value
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($value)
  }];

  let extraClassDeclaration = [{
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable) {
      return IREE::Util::detail::verifyGlobalLoadOp(*this, symbolTable);
    }
  }];
}

class VM_GlobalLoadPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                               list<Trait> traits = []> :
    VM_GlobalLoadOp<type, mnemonic, traits> {
  let description = [{
    Loads the value of a global containing an primitive value.
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncGlobalAttr<"global">,
    VM_EncResult<"value">,
  ];
}

class VM_GlobalStoreOp<Type type, string mnemonic, list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      SymbolUserOpInterface,
      Util_GlobalStoreOpInterface,
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let arguments = (ins
    type:$value,
    VM_GlobalRefAttr:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value)
  }];

  let extraClassDeclaration = [{
    LogicalResult verifySymbolUses(SymbolTableCollection &symbolTable) {
      return IREE::Util::detail::verifyGlobalStoreOp(*this, symbolTable);
    }
  }];
}

class VM_GlobalStorePrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                                list<Trait> traits = []> :
    VM_GlobalStoreOp<type, mnemonic, traits> {
  let description = [{
    Stores a primitive value value to a global.
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncGlobalAttr<"global">,
    VM_EncOperand<"value", 0>,
  ];
}

class VM_GlobalLoadIndirectOp<Type type, string mnemonic,
                              list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      Util_GlobalLoadIndirectOpInterface,
    ])> {
  let arguments = (ins
    AnyTypeOf<[VM_Ptr, Util_PtrOf<type>]>:$global
  );
  let results = (outs
    type:$value
  );

  let assemblyFormat = [{
    $global attr-dict `:` type($global) `->` type($value)
  }];
}

class VM_GlobalLoadIndirectPrimitiveOp<Type type, string mnemonic,
                                       VM_OPC opcode,
                                       list<Trait> traits = []> :
    VM_GlobalLoadIndirectOp<type, mnemonic, traits> {
  let description = [{
    Loads the value of a global containing a primitive value.
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"global", 0>,
    VM_EncResult<"value">,
  ];
}

class VM_GlobalStoreIndirectOp<Type type, string mnemonic,
                               list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      Util_GlobalStoreIndirectOpInterface,
    ])> {
  let arguments = (ins
    type:$value,
    AnyTypeOf<[VM_Ptr, Util_PtrOf<type>]>:$global
  );

  let assemblyFormat = [{
    $value `,` $global attr-dict `:` type($value) `->` type($global)
  }];
}

class VM_GlobalStoreIndirectPrimitiveOp<Type type, string mnemonic,
                                        VM_OPC opcode,
                                        list<Trait> traits = []> :
    VM_GlobalStoreIndirectOp<type, mnemonic, traits> {
  let description = [{
    Stores a primitive value to a global.
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"global", 0>,
    VM_EncOperand<"value", 1>,
  ];
}

def VM_GlobalLoadI32Op :
    VM_GlobalLoadPrimitiveOp<I32, "global.load.i32", VM_OPC_GlobalLoadI32, [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{global 32-bit integer load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadI64Op :
    VM_GlobalLoadPrimitiveOp<I64, "global.load.i64", VM_OPC_GlobalLoadI64, [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{global 64-bit integer load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadF32Op :
    VM_GlobalLoadPrimitiveOp<F32, "global.load.f32", VM_OPC_GlobalLoadF32, [
      VM_ExtF32,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{global 32-bit floating-point load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadF64Op :
    VM_GlobalLoadPrimitiveOp<F64, "global.load.f64", VM_OPC_GlobalLoadF64, [
      VM_ExtF64,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{global 64-bit floating-point load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalStoreI32Op :
    VM_GlobalStorePrimitiveOp<I32, "global.store.i32", VM_OPC_GlobalStoreI32> {
  let summary = [{global 32-bit integer store operation}];
}

def VM_GlobalStoreI64Op :
    VM_GlobalStorePrimitiveOp<I64, "global.store.i64", VM_OPC_GlobalStoreI64> {
  let summary = [{global 64-bit integer store operation}];
}

def VM_GlobalStoreF32Op :
    VM_GlobalStorePrimitiveOp<F32, "global.store.f32", VM_OPC_GlobalStoreF32,
                              [VM_ExtF32]> {
  let summary = [{global 32-bit floating-point store operation}];
}

def VM_GlobalStoreF64Op :
    VM_GlobalStorePrimitiveOp<F64, "global.store.f64", VM_OPC_GlobalStoreF64,
                              [VM_ExtF64]> {
  let summary = [{global 64-bit floating-point store operation}];
}

def VM_GlobalLoadIndirectI32Op :
    VM_GlobalLoadIndirectPrimitiveOp<I32, "global.load.indirect.i32",
                                     VM_OPC_GlobalLoadIndirectI32> {
  let summary = [{global 32-bit integer load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadIndirectI64Op :
    VM_GlobalLoadIndirectPrimitiveOp<I64, "global.load.indirect.i64",
                                     VM_OPC_GlobalLoadIndirectI64> {
  let summary = [{global 64-bit integer load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadIndirectF32Op :
    VM_GlobalLoadIndirectPrimitiveOp<F32, "global.load.indirect.f32",
                                     VM_OPC_GlobalLoadIndirectF32,
                                     [VM_ExtF64]> {
  let summary = [{global 32-bit floating-point load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadIndirectF64Op :
    VM_GlobalLoadIndirectPrimitiveOp<F64, "global.load.indirect.f64",
                                     VM_OPC_GlobalLoadIndirectF64,
                                     [VM_ExtF64]> {
  let summary = [{global 64-bit floating-point load operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalStoreIndirectI32Op :
    VM_GlobalStoreIndirectPrimitiveOp<I32, "global.store.indirect.i32",
                                      VM_OPC_GlobalStoreIndirectI32> {
  let summary = [{global 32-bit integer store operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalStoreIndirectI64Op :
    VM_GlobalStoreIndirectPrimitiveOp<I64, "global.store.indirect.i64",
                                      VM_OPC_GlobalStoreIndirectI64> {
  let summary = [{global 64-bit integer store operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalStoreIndirectF32Op :
    VM_GlobalStoreIndirectPrimitiveOp<F32, "global.store.indirect.f32",
                                      VM_OPC_GlobalStoreIndirectI32,
                                      [VM_ExtF32]> {
  let summary = [{global 32-bit floating-point store operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalStoreIndirectF64Op :
    VM_GlobalStoreIndirectPrimitiveOp<F64, "global.store.indirect.f64",
                                      VM_OPC_GlobalStoreIndirectF64,
                                      [VM_ExtF64]> {
  let summary = [{global 64-bit floating-point store operation}];
  let hasCanonicalizer = 1;
}

def VM_GlobalLoadRefOp : VM_GlobalLoadOp<VM_AnyRef, "global.load.ref", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{global ref<T> load operation}];
  let description = [{
    Loads the value of a global containing a ref of the given type.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalLoadRef>,
    VM_EncGlobalAttr<"global">,
    VM_EncTypeOf<"value">,
    VM_EncResult<"value">,
  ];
}

def VM_GlobalStoreRefOp : VM_GlobalStoreOp<VM_AnyRef, "global.store.ref"> {
  let summary = [{global ref<T> stores operation}];
  let description = [{
    Stores a ref<T> to a global, retaining it until the global is reset.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalStoreRef>,
    VM_EncGlobalAttr<"global">,
    VM_EncTypeOf<"value">,
    VM_EncOperand<"value", 0>,
  ];
}

def VM_GlobalLoadIndirectRefOp :
    VM_GlobalLoadIndirectOp<VM_AnyRef, "global.load.indirect.ref"> {
  let summary = [{global ref<T> load operation}];
  let description = [{
    Loads the value of a global containing a ref of the given type.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalLoadIndirectRef>,
    VM_EncOperand<"global", 0>,
    VM_EncTypeOf<"value">,
    VM_EncResult<"value">,
  ];

  let hasCanonicalizer = 1;
}

def VM_GlobalStoreIndirectRefOp :
    VM_GlobalStoreIndirectOp<VM_AnyRef, "global.store.indirect.ref"> {
  let summary = [{global ref<T> stores operation}];
  let description = [{
    Stores a ref<T> to a global, retaining it until the global is reset.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalStoreIndirectRef>,
    VM_EncOperand<"global", 0>,
    VM_EncTypeOf<"value">,
    VM_EncOperand<"value", 1>,
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

class VM_ConstOp<string mnemonic, string ctype, list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      ConstantLike,
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypedAttr":$value)>,
    OpBuilder<(ins ctype:$value)>,
  ];

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(TypedAttr value, Type type);

    /// Returns an attribute in the appropriate type for the const op.
    static TypedAttr convertConstValue(TypedAttr value);
  }];
}

class VM_ConstantPrimitiveOp<Type type, int width, string mnemonic,
                             VM_OPC opcode,
                             string ctype, list<Trait> traits = []> :
    VM_ConstOp<mnemonic, ctype, traits> {
  let description = [{
    Defines a constant value that is treated as a scalar literal at runtime.
  }];

  let results = (outs
    type:$result
  );

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncPrimitiveAttr<"value", width>,
    VM_EncResult<"result">,
  ];

  let assemblyFormat = [{
    $value attr-dict
  }];
}

def VM_ConstI32Op :
    VM_ConstantPrimitiveOp<I32, 32, "const.i32", VM_OPC_ConstI32,
                           "int32_t", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{32-bit integer constant operation}];
  let arguments = (ins
    VM_ConstantIntegerValueAttr<I32>:$value
  );
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_ConstI64Op :
    VM_ConstantPrimitiveOp<I64, 64, "const.i64", VM_OPC_ConstI64,
                           "int64_t", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{64-bit integer constant operation}];
  let arguments = (ins
    VM_ConstantIntegerValueAttr<I64>:$value
  );
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_ConstF32Op :
    VM_ConstantPrimitiveOp<F32, 32, "const.f32", VM_OPC_ConstF32,
                           "float", [VM_ExtF32]> {
  let summary = [{32-bit floating-point constant operation}];
  let arguments = (ins
    VM_ConstantFloatValueAttr<F32>:$value
  );
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_ConstF64Op :
    VM_ConstantPrimitiveOp<F64, 64, "const.f64", VM_OPC_ConstF64,
                           "double", [VM_ExtF64]> {
  let summary = [{64-bit floating-point constant operation}];
  let arguments = (ins
    VM_ConstantFloatValueAttr<F64>:$value
  );
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

class VM_ConstantPrimitiveZeroOp<Type type, string mnemonic, VM_OPC opcode,
                                 string ctype, list<Trait> traits = []> :
    VM_ConstOp<mnemonic, ctype, traits> {
  let description = [{
    Defines a constant zero primitive.
  }];

  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    attr-dict
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncResult<"result">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins)>,
  ];
}

def VM_ConstI32ZeroOp :
    VM_ConstantPrimitiveZeroOp<I32, "const.i32.zero", VM_OPC_ConstI32Zero,
                               "int32_t", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{32-bit integer constant zero operation}];
  let hasFolder = 1;
}

def VM_ConstI64ZeroOp :
    VM_ConstantPrimitiveZeroOp<I64, "const.i64.zero", VM_OPC_ConstI64Zero,
                               "int64_t", [
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{64-bit integer constant zero operation}];
  let hasFolder = 1;
}

def VM_ConstF32ZeroOp :
    VM_ConstantPrimitiveZeroOp<F32, "const.f32.zero", VM_OPC_ConstF32Zero,
                               "float", [
      VM_ExtF32,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{32-bit floating-point constant zero operation}];
  let hasFolder = 1;
}

def VM_ConstF64ZeroOp :
    VM_ConstantPrimitiveZeroOp<F64, "const.f64.zero", VM_OPC_ConstF64Zero,
                               "double", [
      VM_ExtF64,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{64-bit floating-point constant zero operation}];
  let hasFolder = 1;
}

def VM_ConstRefZeroOp : VM_PureOp<"const.ref.zero", [
    ConstantLike,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = [{null ref constant operation}];
  let description = [{
    Defines a constant null ref that can be used in comparisons and
    initialization.
  }];

  let results = (outs
    VM_AnyRef:$result
  );

  let assemblyFormat = [{
    `:` type($result) attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ConstRefZero>,
    VM_EncResult<"result">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$objectType)>,
  ];

  let hasFolder = 1;
}

def VM_RodataOp : VM_Op<"rodata", [
    IsolatedFromAbove,
    HasParent<"IREE::VM::ModuleOp">,
    Symbol,
  ]> {
  let summary = [{read-only data definition operation}];
  let description = [{
    Defines a blob of read-only constant data that can be represented as a
    ref. This can be used to store arbitrary data within modules such as
    large constant buffers and other file contents.

    Note that the data is reference counted as a way to track its usage once the
    value leaves the module. For example, returning rodata from an exported
    function must keep the data (possibly backed by mmap) valid for its entire
    lifetime.

    By default all rodata will be aligned in the final module output at a
    16-byte granularity. An optional alignment can be specified to override the
    default for cases where larger or smaller alignments are needed.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$sym_visibility,
    SymbolNameAttr:$sym_name,
    Util_AnySerializableAttr:$value,
    OptionalAttr<I64Attr>:$alignment,
    OptionalAttr<VM_Ordinal>:$ordinal,
    OptionalAttr<StrAttr>:$mime_type
  );

  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) $sym_name attr-dict $value
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
      "StringRef":$name,
      "Attribute":$value,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
    )>,
  ];
}

def VM_ConstRefRodataOp : VM_PureOp<"const.ref.rodata", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = [{constant rodata access operation}];
  let description = [{
    Returns a reference to a read-only buffer.
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$rodata
  );

  let results = (outs
    VM_RefOf<VM_BufferType>:$value
  );

  let assemblyFormat = [{
    $rodata attr-dict `:` type($value)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ConstRefRodata>,
    VM_EncRodataAttr<"rodata">,
    VM_EncResult<"value">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$rodataName,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    OpBuilder<(ins "RodataOp":$rodataOp,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
  ];

  let hasVerifier = 1;
}

def VM_RodataInlineOp : VM_PureOp<"rodata.inline", [
    VM_PseudoOp,
  ]> {
  let summary = [{inlined constant rodata}];
  let description = [{
    vm.rodata that can be embedded inline in functions. See vm.rodata for more
    information.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr>:$name,
    Util_AnySerializableAttr:$value,
    OptionalAttr<I64Attr>:$alignment,
    OptionalAttr<StrAttr>:$mime_type
  );

  let results = (outs
    VM_RefOf<VM_BufferType>:$result
  );

  let assemblyFormat = [{
    ($name^)? attr-dict `:` type($result) `=` $value
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Attribute":$value),
    [{
      $_state.addTypes({resultType});
      $_state.addAttribute("value", value);
    }]>,
    OpBuilder<(ins "StringAttr":$value)>
  ];
}

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

def VM_BufferAllocOp :
    VM_Op<"buffer.alloc", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemAlloc]>,
      AlwaysSpeculatable,
    ]> {
  let summary = [{allocates a new zero-initialized buffer}];
  let description = [{
    Allocates a new zero-initialized buffer with the given size in bytes.
  }];

  let arguments = (ins
    VM_BufferIndex:$length
  );
  let results = (outs
    VM_RefOf<VM_BufferType>:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_BufferAlloc>,
    VM_EncOperand<"length", 0>,
    VM_EncResult<"result">,
  ];
}

def VM_BufferCloneOp :
    VM_Op<"buffer.clone", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemAlloc, MemRead]>,
      AlwaysSpeculatable,
    ]> {
  let summary = [{clones a buffer}];
  let description = [{
    Clones a range of the source buffer to produce a mutable buffer with the
    same contents.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$source_buffer,
    VM_BufferIndex:$source_offset,
    VM_BufferIndex:$length
  );
  let results = (outs
    VM_RefOf<VM_BufferType>:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($source_buffer) `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_BufferClone>,
    VM_EncOperand<"source_buffer", 0>,
    VM_EncOperand<"source_offset", 1>,
    VM_EncOperand<"length", 2>,
    VM_EncResult<"result">,
  ];
}

def VM_BufferLengthOp :
    VM_PureOp<"buffer.length", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{returns the byte length of a buffer}];
  let description = [{
    Returns the total byte length of the given buffer. This is the exact value
    as specified during buffer allocation though the underlying system buffer
    may have additional padding.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$buffer
  );
  let results = (outs
    VM_BufferIndex:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($buffer) `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_BufferLength>,
    VM_EncOperand<"buffer", 0>,
    VM_EncResult<"result">,
  ];
}

def VM_BufferCopyOp :
    VM_Op<"buffer.copy", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemRead, MemWrite]>,
    ]> {
  let summary = [{copies a range of a buffer to another}];
  let description = [{
    Copies a range of one buffer to another, like memcpy.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$source_buffer,
    VM_BufferIndex:$source_offset,
    VM_RefOf<VM_BufferType>:$target_buffer,
    VM_BufferIndex:$target_offset,
    VM_BufferIndex:$length
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($source_buffer) `->` type($target_buffer)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_BufferCopy>,
    VM_EncOperand<"source_buffer", 0>,
    VM_EncOperand<"source_offset", 1>,
    VM_EncOperand<"target_buffer", 2>,
    VM_EncOperand<"target_offset", 3>,
    VM_EncOperand<"length", 4>
  ];
}

def VM_BufferCompareOp :
    VM_Op<"buffer.compare", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemRead, MemWrite]>,
    ]> {
  let summary = [{compares a range of a buffer to another}];
  let description = [{
    Returns 1 if the two ranges are bitwise equivalent, somewhat like memcmp.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$lhs_buffer,
    VM_BufferIndex:$lhs_offset,
    VM_RefOf<VM_BufferType>:$rhs_buffer,
    VM_BufferIndex:$rhs_offset,
    VM_BufferIndex:$length
  );
  let results = (outs
    VM_CondValue:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($lhs_buffer) `,` type($rhs_buffer)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_BufferCompare>,
    VM_EncOperand<"lhs_buffer", 0>,
    VM_EncOperand<"lhs_offset", 1>,
    VM_EncOperand<"rhs_buffer", 2>,
    VM_EncOperand<"rhs_offset", 3>,
    VM_EncOperand<"length", 4>,
    VM_EncResult<"result">
  ];
}

class VM_BufferFillOp<Type type, string mnemonic, VM_OPC opcode,
                      list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemWrite]>,
    ])> {
  let description = [{
    Fills a range of the buffer with the given value, like memset.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$target_buffer,
    VM_BufferIndex:$target_offset,
    VM_BufferIndex:$length,
    AnyTypeOf<[type, I32]>:$value
  );

  let assemblyFormat = [{
    $target_buffer `,` $target_offset `,` $length `,` $value
    attr-dict `:` type($value) `->` type($target_buffer)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"target_buffer", 0>,
    VM_EncOperand<"target_offset", 1>,
    VM_EncOperand<"length", 2>,
    VM_EncOperand<"value", 3>
  ];
}

def VM_BufferFillI8Op :
    VM_BufferFillOp<I8, "buffer.fill.i8", VM_OPC_BufferFillI8> {
  let summary = [{fills the buffer with the given repeating 8-bit value}];
}
def VM_BufferFillI16Op :
    VM_BufferFillOp<I16, "buffer.fill.i16", VM_OPC_BufferFillI16> {
  let summary = [{fills the buffer with the given repeating 16-bit value}];
}
def VM_BufferFillI32Op :
    VM_BufferFillOp<I32, "buffer.fill.i32", VM_OPC_BufferFillI32> {
  let summary = [{fills the buffer with the given repeating 32-bit value}];
}
def VM_BufferFillI64Op :
    VM_BufferFillOp<I64, "buffer.fill.i64", VM_OPC_BufferFillI64> {
  let summary = [{fills the buffer with the given repeating 64-bit value}];
}
def VM_BufferFillF32Op :
    VM_BufferFillOp<F32, "buffer.fill.f32", VM_OPC_BufferFillF32,
                    [VM_ExtF32]> {
  let summary = [{fills the buffer with the given repeating 32-bit value}];
}
def VM_BufferFillF64Op :
    VM_BufferFillOp<F64, "buffer.fill.f64", VM_OPC_BufferFillF64,
                    [VM_ExtF64]> {
  let summary = [{fills the buffer with the given repeating 64-bit value}];
}

class VM_BufferLoadOp<Type type, string mnemonic, VM_OPC opcode,
                      list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemRead]>,
    ])> {
  let description = [{
    Loads a value from the buffer at the given offset. Offsets must match the
    natural alignment of the loaded type and will be coerced to the alignment
    at runtime automatically.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$source_buffer,
    VM_BufferIndex:$source_offset
  );
  let results = (outs
    AnyTypeOf<[type, I32]>:$result
  );

  let assemblyFormat = [{
    $source_buffer `[` $source_offset `]`
    attr-dict `:` type($source_buffer) `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"source_buffer", 0>,
    VM_EncOperand<"source_offset", 1>,
    VM_EncResult<"result">,
  ];
}

class VM_BufferStoreOp<Type type, string mnemonic, VM_OPC opcode,
                       list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemWrite]>,
    ])> {
  let description = [{
    Stores a value to the buffer at the given offset. Offsets must match the
    natural alignment of the loaded type and will be coerced to the alignment
    at runtime automatically.
  }];

  let arguments = (ins
    VM_RefOf<VM_BufferType>:$target_buffer,
    VM_BufferIndex:$target_offset,
    AnyTypeOf<[type, I32]>:$value
  );

  let assemblyFormat = [{
    $value `,` $target_buffer `[` $target_offset `]`
    attr-dict `:` type($value) `->` type($target_buffer)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"target_buffer", 0>,
    VM_EncOperand<"target_offset", 1>,
    VM_EncOperand<"value", 2>,
  ];
}

def VM_BufferLoadI8UOp :
    VM_BufferLoadOp<I32, "buffer.load.i8.u", VM_OPC_BufferLoadI8U, []> {
  let summary = [{unsigned 8-bit integer load}];
}
def VM_BufferLoadI8SOp :
    VM_BufferLoadOp<I32, "buffer.load.i8.s", VM_OPC_BufferLoadI8S, []> {
  let summary = [{signed 8-bit integer load}];
}

def VM_BufferLoadI16UOp :
    VM_BufferLoadOp<I32, "buffer.load.i16.u", VM_OPC_BufferLoadI16U, []> {
  let summary = [{unsigned 16-bit integer load}];
}
def VM_BufferLoadI16SOp :
    VM_BufferLoadOp<I32, "buffer.load.i16.s", VM_OPC_BufferLoadI16S, []> {
  let summary = [{signed 16-bit integer load}];
}

def VM_BufferLoadI32Op :
    VM_BufferLoadOp<I32, "buffer.load.i32", VM_OPC_BufferLoadI32, []> {
  let summary = [{32-bit integer load}];
}
def VM_BufferLoadI64Op :
    VM_BufferLoadOp<I64, "buffer.load.i64", VM_OPC_BufferLoadI64> {
  let summary = [{64-bit integer load}];
}
def VM_BufferLoadF32Op :
    VM_BufferLoadOp<F32, "buffer.load.f32", VM_OPC_BufferLoadF32,
                    [VM_ExtF32]> {
  let summary = [{32-bit floating-point load}];
}
def VM_BufferLoadF64Op :
    VM_BufferLoadOp<F64, "buffer.load.f64", VM_OPC_BufferLoadF64,
                    [VM_ExtF64]> {
  let summary = [{64-bit floating-point load}];
}

def VM_BufferStoreI8Op :
    VM_BufferStoreOp<I32, "buffer.store.i8", VM_OPC_BufferStoreI8, []> {
  let summary = [{unsigned 8-bit integer store}];
}
def VM_BufferStoreI16Op :
    VM_BufferStoreOp<I32, "buffer.store.i16", VM_OPC_BufferStoreI16, []> {
  let summary = [{unsigned 16-bit integer store}];
}
def VM_BufferStoreI32Op :
    VM_BufferStoreOp<I32, "buffer.store.i32", VM_OPC_BufferStoreI32, []> {
  let summary = [{32-bit integer store}];
}
def VM_BufferStoreI64Op :
    VM_BufferStoreOp<I64, "buffer.store.i64", VM_OPC_BufferStoreI64> {
  let summary = [{64-bit integer store}];
}
def VM_BufferStoreF32Op :
    VM_BufferStoreOp<F32, "buffer.store.f32", VM_OPC_BufferStoreF32,
                     [VM_ExtF32]> {
  let summary = [{32-bit floating-point store}];
}
def VM_BufferStoreF64Op :
    VM_BufferStoreOp<F64, "buffer.store.f64", VM_OPC_BufferStoreF64,
                     [VM_ExtF64]> {
  let summary = [{64-bit floating-point store}];
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

// TODO(benvanik): vm.list.push.i32 / vm.list.pop.i32 (variadic)
// TODO(benvanik): vm.list.copy(src_list, src_index, dst_list, dst_index, length)
// TODO(benvanik): vm.list.slice(list, index, length) -> list
// TODO(benvanik): export into a rwdata buffer

def VM_ListAllocOp :
    VM_Op<"list.alloc", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemAlloc]>,
    ]> {
  let summary = [{allocates a new empty list}];
  let description = [{
    Allocates a new typed list with a minimum initial_capacity.
  }];

  let arguments = (ins
    VM_ListIndex:$initial_capacity
  );
  let results = (outs
    VM_AnyList:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($initial_capacity) `)` `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ListAlloc>,
    VM_EncType<"getResult().getType().cast<IREE::VM::RefType>().getObjectType().cast<IREE::VM::ListType>().getElementType()">,
    VM_EncOperand<"initial_capacity", 0>,
    VM_EncResult<"result">,
  ];
}

def VM_ListReserveOp :
    VM_Op<"list.reserve", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemAlloc, MemRead, MemWrite]>,
    ]> {
  let summary = [{reserves capacity for list growth}];
  let description = [{
    Reserves storage for at least minimum_capacity elements. If the list already
    has at least the specified capacity the operation is ignored.
  }];

  let arguments = (ins
    VM_AnyList:$list,
    VM_ListIndex:$minimum_capacity
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `,` type($minimum_capacity) `)`
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ListReserve>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"minimum_capacity", 1>,
  ];
}

def VM_ListSizeOp :
    VM_Op<"list.size", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemRead]>,
    ]> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    VM_AnyList:$list
  );
  let results = (outs
    VM_ListIndex:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `)` `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ListSize>,
    VM_EncOperand<"list", 0>,
    VM_EncResult<"result">,
  ];
}

def VM_ListResizeOp :
    VM_Op<"list.resize", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemWrite]>,
    ]> {
  let summary = [{resizes the list to a new count in elements}];
  let description = [{
    Resizes the list to contain new_size elements. This will either truncate
    the list if the existing size is greater than new_size or extend the list
    with the default list value of 0 if storing primitives and null if refs.
  }];

  let arguments = (ins
    VM_AnyList:$list,
    VM_ListIndex:$new_size
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `,` type($new_size) `)`
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ListResize>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"new_size", 1>,
  ];
}

class VM_ListGetPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                            list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemRead]>,
    ])> {
  let summary = [{primitive type element accessor}];
  let description = [{
    Returns the value of the element at the given index.
  }];

  let arguments = (ins
    VM_ListOf<VM_PrimitiveType>:$list,
    VM_ListIndex:$index
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `,` type($index) `)` `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncResult<"result">,
  ];
}

class VM_ListSetPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                            list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemWrite]>,
    ])> {
  let summary = [{primitive type element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    VM_ListOf<VM_PrimitiveType>:$list,
    VM_ListIndex:$index,
    type:$value
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `,` type($index) `,` type($value) `)`
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncOperand<"value", 2>,
  ];
}

def VM_ListGetI32Op :
    VM_ListGetPrimitiveOp<I32, "list.get.i32", VM_OPC_ListGetI32>;

def VM_ListGetI64Op :
    VM_ListGetPrimitiveOp<I64, "list.get.i64", VM_OPC_ListGetI64>;

def VM_ListGetF32Op :
    VM_ListGetPrimitiveOp<F32, "list.get.f32", VM_OPC_ListGetF32, [VM_ExtF32]>;

def VM_ListGetF64Op :
    VM_ListGetPrimitiveOp<F64, "list.get.f64", VM_OPC_ListGetF64, [VM_ExtF64]>;

def VM_ListSetI32Op :
    VM_ListSetPrimitiveOp<I32, "list.set.i32", VM_OPC_ListSetI32>;

def VM_ListSetI64Op :
    VM_ListSetPrimitiveOp<I64, "list.set.i64", VM_OPC_ListSetI64>;

def VM_ListSetF32Op :
    VM_ListSetPrimitiveOp<F32, "list.set.f32", VM_OPC_ListSetF32, [VM_ExtF32]>;

def VM_ListSetF64Op :
    VM_ListSetPrimitiveOp<F64, "list.set.f64", VM_OPC_ListSetF64, [VM_ExtF64]>;

def VM_ListGetRefOp :
    VM_Op<"list.get.ref", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemRead]>,
    ]> {
  let summary = [{ref type element accessor}];
  let description = [{
    Returns the ref value of the element at the given index. Note that the value
    may be null if the element is null or the type does not match.
  }];

  let arguments = (ins
    VM_AnyList:$list,
    VM_ListIndex:$index
  );
  let results = (outs
    VM_AnyRef:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `,` type($index) `)` `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ListGetRef>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncTypeOf<"result">,
    VM_EncResult<"result">,
  ];

  let hasVerifier = 1;
}

def VM_ListSetRefOp :
    VM_Op<"list.set.ref", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      MemoryEffects<[MemWrite]>,
    ]> {
  let summary = [{ref type element mutator}];
  let description = [{
    Sets the element at the given index to the new ref value (possibly null).
  }];

  let arguments = (ins
    VM_AnyList:$list,
    VM_ListIndex:$index,
    VM_AnyRef:$value
  );

  let assemblyFormat = [{
    operands attr-dict `:` `(` type($list) `,` type($index) `,` type($value) `)`
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ListSetRef>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncOperand<"value", 2>,
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Conditional assignment
//===----------------------------------------------------------------------===//

class VM_SelectPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                           list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["true_value", "false_value", "result"]>,
    ])> {
  let description = [{
    Chooses one value based on a binary condition supplied as its first operand.
    If the value of the condition is true the `true_value` operand is chosen,
    otherwise the `false_value` operand is chosen. The true and false values
    must have the same types. For example, the maximum operation is obtained by
    combining "select" with "cmpi" as follows:

    ```
    %2 = vm.cmp.gt.i32.s %0, %1 : i32
    %3 = vm.select.i32 %2, %0, %1 : i32
    ```
  }];

  let arguments = (ins
    VM_CondValue:$condition,
    type:$true_value,
    type:$false_value
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"condition", 0>,
    VM_EncOperand<"true_value", 1>,
    VM_EncOperand<"false_value", 2>,
    VM_EncResult<"result">,
  ];
}

def VM_SelectI32Op : VM_SelectPrimitiveOp<I32, "select.i32", VM_OPC_SelectI32> {
  let summary = [{integer select operation}];
  let hasFolder = 1;
}

def VM_SelectI64Op : VM_SelectPrimitiveOp<I64, "select.i64", VM_OPC_SelectI64> {
  let summary = [{integer select operation}];
  let hasFolder = 1;
}

def VM_SelectF32Op : VM_SelectPrimitiveOp<F32, "select.f32", VM_OPC_SelectF32,
                                          [VM_ExtF32]> {
  let summary = [{floating-point select operation}];
  let hasFolder = 1;
}

def VM_SelectF64Op : VM_SelectPrimitiveOp<F64, "select.f64", VM_OPC_SelectF64,
                                          [VM_ExtF64]> {
  let summary = [{floating-point select operation}];
  let hasFolder = 1;
}

def VM_SelectRefOp : VM_PureOp<"select.ref", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    AllTypesMatch<["true_value", "false_value", "result"]>,
  ]> {
  let summary = [{ref<T> select operation}];
  let description = [{
    Chooses one value based on a binary condition supplied as its first operand.
    If the value of the condition is true the `true_value` operand is chosen,
    otherwise the `false_value` operand is chosen.
  }];

  let arguments = (ins
    VM_CondValue:$condition,
    VM_AnyRef:$true_value,
    VM_AnyRef:$false_value
  );
  let results = (outs
    VM_AnyRef:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_SelectRef>,
    VM_EncOperand<"condition", 0>,
    VM_EncTypeOf<"result">,
    VM_EncOperand<"true_value", 1>,
    VM_EncOperand<"false_value", 2>,
    VM_EncResult<"result">,
  ];

  let hasFolder = 1;
}

class VM_SwitchIntegerOp<I type, string mnemonic, VM_OPC opcode,
                         list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["default_value", "result"]>,
    ])> {
  let description = [{
    Returns the value with the given `index` in `values` or `default_value` if
    the index is out of bounds.

    ```mlir
    // Switch %index to cases of %c100/%c200/%c300 if index==0, ==1, ==2.
    // If %index is out of range (<0 or >2) then default to %c5.
    %0 = vm.switch.i32 %index[%c100, %c200, %c300] else %c5 : i32
    ```
  }];

  let arguments = (ins
    VM_Index:$index,
    type:$default_value,
    Variadic<type>:$values
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    $index `[` $values `]` `else` $default_value attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"index", 0>,
    VM_EncPrimitiveAttr<"default_value", type.bitwidth>,
    VM_EncVariadicOperands<"values">,
    VM_EncResult<"result">,
  ];
}

class VM_SwitchFloatOp<F type, string mnemonic, VM_OPC opcode,
                       list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["default_value", "result"]>,
    ])> {
  let description = [{
    Returns the value with the given `index` in `values` or `default_value` if
    the index is out of bounds.

    ```mlir
    // Switch %index to cases of %c100/%c200/%c300 if index==0, ==1, ==2.
    // If %index is out of range (<0 or >2) then default to %c5.
    %0 = vm.switch.f32 %index[%c100, %c200, %c300] else %c5 : f32
    ```
  }];

  let arguments = (ins
    VM_Index:$index,
    type:$default_value,
    Variadic<type>:$values
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    $index `[` $values `]` `else` $default_value attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"index", 0>,
    VM_EncPrimitiveAttr<"default_value", type.bitwidth>,
    VM_EncVariadicOperands<"values">,
    VM_EncResult<"result">,
  ];
}

def VM_SwitchI32Op : VM_SwitchIntegerOp<I32, "switch.i32", VM_OPC_SwitchI32> {
  let summary = [{integer switch operation}];
  let hasFolder = 1;
}

def VM_SwitchI64Op : VM_SwitchIntegerOp<I64, "switch.i64", VM_OPC_SwitchI64> {
  let summary = [{integer switch operation}];
  let hasFolder = 1;
}

def VM_SwitchF32Op : VM_SwitchFloatOp<F32, "switch.f32", VM_OPC_SwitchF32,
                                      [VM_ExtF32]> {
  let summary = [{floating-point switch operation}];
  let hasFolder = 1;
}

def VM_SwitchF64Op : VM_SwitchFloatOp<F64, "switch.f64", VM_OPC_SwitchF64,
                                      [VM_ExtF64]> {
  let summary = [{floating-point switch operation}];
  let hasFolder = 1;
}

def VM_SwitchRefOp : VM_PureOp<"switch.ref", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    AllTypesMatch<["default_value", "result"]>,
  ]> {
  let summary = [{ref<T> switch operation}];
  let description = [{
    Returns the value with the given `index` in `values` or `default_value` if
    the index is out of bounds.

    ```mlir
    // Switch %arg0 to cases of %r0/%r1/%r2 if arg0==0, ==1, ==2.
    // If %arg0 is out of range (<0 or >2) then default to %null.
    %0 = vm.switch.ref %index[%r0, %r1, %r2] else %null : vm.ref<!foo>
    ```
  }];

  let arguments = (ins
    VM_Index:$index,
    VM_AnyRef:$default_value,
    Variadic<VM_AnyRef>:$values
  );
  let results = (outs
    VM_AnyRef:$result
  );

  // TODO(benvanik): figure out how to get $values types in the format.
  // let assemblyFormat = [{
  //   $index `[` $values `]` `else` $default_value attr-dict `:` type($values)
  // }];

  let encoding = [
    VM_EncOpcode<VM_OPC_SwitchRef>,
    VM_EncOperand<"index", 0>,
    VM_EncTypeOf<"result">,
    VM_EncOperand<"default_value", 1>,
    VM_EncVariadicOperands<"values">,
    VM_EncResult<"result">,
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Arithmetic helpers
//===----------------------------------------------------------------------===//

class VM_UnaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                           list<Trait> traits = []> :
    VM_TrivialOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["operand", "result"]>,
    ])> {
  let arguments = (ins
    type:$operand
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncResult<"result">,
  ];
}

class VM_BinaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                            list<Trait> traits = []> :
    VM_TrivialOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["lhs", "rhs", "result"]>,
    ])> {
  let arguments = (ins
    type:$lhs,
    type:$rhs
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"lhs", 0>,
    VM_EncOperand<"rhs", 1>,
    VM_EncResult<"result">,
  ];
}

class VM_TernaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                             list<Trait> traits = []> :
    VM_TrivialOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["a", "b", "c", "result"]>,
    ])> {
  let arguments = (ins
    type:$a,
    type:$b,
    type:$c
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"a", 0>,
    VM_EncOperand<"b", 1>,
    VM_EncOperand<"c", 2>,
    VM_EncResult<"result">,
  ];
}

// Arithmetic ops defined for all inputs which do not exhibit UB.
class VM_TotalUnaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                                list<Trait> traits = []> :
    VM_UnaryArithmeticOp<type, mnemonic, opcode,
                         !listconcat(traits, [AlwaysSpeculatable])> {}

class VM_TotalBinaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                                 list<Trait> traits = []> :
    VM_BinaryArithmeticOp<type, mnemonic, opcode,
                          !listconcat(traits, [AlwaysSpeculatable])> {}

class VM_TotalTernaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                                  list<Trait> traits = []> :
    VM_TernaryArithmeticOp<type, mnemonic, opcode,
                           !listconcat(traits, [AlwaysSpeculatable])> {}

//===----------------------------------------------------------------------===//
// Integer arithmetic
//===----------------------------------------------------------------------===//

def VM_AddI32Op :
    VM_BinaryArithmeticOp<I32, "add.i32", VM_OPC_AddI32, [Commutative]> {
  let summary = [{integer add operation}];
  let hasFolder = 1;
}

def VM_AddI64Op :
    VM_BinaryArithmeticOp<I64, "add.i64", VM_OPC_AddI64, [Commutative]> {
  let summary = [{integer add operation}];
  let hasFolder = 1;
}

def VM_SubI32Op :
    VM_BinaryArithmeticOp<I32, "sub.i32", VM_OPC_SubI32> {
  let summary = [{integer subtract operation}];
  let hasFolder = 1;
}

def VM_SubI64Op :
    VM_BinaryArithmeticOp<I64, "sub.i64", VM_OPC_SubI64> {
  let summary = [{integer subtract operation}];
  let hasFolder = 1;
}

def VM_MulI32Op :
    VM_BinaryArithmeticOp<I32, "mul.i32", VM_OPC_MulI32, [Commutative]> {
  let summary = [{integer multiplication operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_MulI64Op :
    VM_BinaryArithmeticOp<I64, "mul.i64", VM_OPC_MulI64, [Commutative]> {
  let summary = [{integer multiplication operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_DivI32SOp :
    VM_BinaryArithmeticOp<I32, "div.i32.s", VM_OPC_DivI32S> {
  let summary = [{signed integer division operation}];
  let hasFolder = 1;
}

def VM_DivI64SOp :
    VM_BinaryArithmeticOp<I64, "div.i64.s", VM_OPC_DivI64S> {
  let summary = [{signed integer division operation}];
  let hasFolder = 1;
}

def VM_DivI32UOp :
    VM_BinaryArithmeticOp<I32, "div.i32.u", VM_OPC_DivI32U> {
  let summary = [{unsigned integer division operation}];
  let hasFolder = 1;
}

def VM_DivI64UOp :
    VM_BinaryArithmeticOp<I64, "div.i64.u", VM_OPC_DivI64U> {
  let summary = [{unsigned integer division operation}];
  let hasFolder = 1;
}

def VM_RemI32SOp :
    VM_BinaryArithmeticOp<I32, "rem.i32.s", VM_OPC_RemI32S> {
  let summary = [{signed integer division remainder operation}];
  let hasFolder = 1;
}

def VM_RemI64SOp :
    VM_BinaryArithmeticOp<I64, "rem.i64.s", VM_OPC_RemI64S> {
  let summary = [{signed integer division remainder operation}];
  let hasFolder = 1;
}

def VM_RemI32UOp :
    VM_BinaryArithmeticOp<I32, "rem.i32.u", VM_OPC_RemI32U> {
  let summary = [{unsigned integer division remainder operation}];
  let hasFolder = 1;
}

def VM_RemI64UOp :
    VM_BinaryArithmeticOp<I64, "rem.i64.u", VM_OPC_RemI64U> {
  let summary = [{unsigned integer division remainder operation}];
  let hasFolder = 1;
}

def VM_FMAI32Op :
    VM_TernaryArithmeticOp<I32, "fma.i32", VM_OPC_FMAI32> {
  let summary = [{integer fused-multiply add operation (a*b+c)}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_FMAI64Op :
    VM_TernaryArithmeticOp<I64, "fma.i64", VM_OPC_FMAI64> {
  let summary = [{integer fused-multiply add operation (a*b+c)}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_AbsI32Op :
    VM_UnaryArithmeticOp<I32, "abs.i32", VM_OPC_AbsI32> {
  let summary = [{integer absolute-value operation}];
  let hasFolder = 1;
}

def VM_AbsI64Op :
    VM_UnaryArithmeticOp<I64, "abs.i64", VM_OPC_AbsI64> {
  let summary = [{integer absolute-value operation}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Floating-point arithmetic
//===----------------------------------------------------------------------===//

def VM_AddF32Op :
    VM_TotalBinaryArithmeticOp<F32, "add.f32", VM_OPC_AddF32,
                               [VM_ExtF32, Commutative]> {
  let summary = [{floating-point add operation}];
  let hasFolder = 1;
}

def VM_AddF64Op :
    VM_TotalBinaryArithmeticOp<F64, "add.f64", VM_OPC_AddF64,
                               [VM_ExtF64, Commutative]> {
  let summary = [{floating-point add operation}];
  let hasFolder = 1;
}

def VM_SubF32Op :
    VM_TotalBinaryArithmeticOp<F32, "sub.f32", VM_OPC_SubF32,
                              [VM_ExtF32]> {
  let summary = [{floating point subtraction operation}];
  let hasFolder = 1;
}

def VM_SubF64Op :
    VM_TotalBinaryArithmeticOp<F64, "sub.f64", VM_OPC_SubF64,
                               [VM_ExtF64]> {
  let summary = [{floating point subtraction operation}];
  let hasFolder = 1;
}

def VM_MulF32Op :
    VM_TotalBinaryArithmeticOp<F32, "mul.f32", VM_OPC_MulF32,
                               [VM_ExtF32, Commutative]> {
  let summary = [{floating point multiplication operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_MulF64Op :
    VM_TotalBinaryArithmeticOp<F64, "mul.f64", VM_OPC_MulF64,
                               [VM_ExtF64, Commutative]> {
  let summary = [{floating point multiplication operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_DivF32Op :
    VM_TotalBinaryArithmeticOp<F32, "div.f32", VM_OPC_DivF32,
                               [VM_ExtF32]> {
  let summary = [{floating point division operation}];
  let hasFolder = 1;
}

def VM_DivF64Op :
    VM_TotalBinaryArithmeticOp<F64, "div.f64", VM_OPC_DivF64,
                               [VM_ExtF64]> {
  let summary = [{floating point division operation}];
  let hasFolder = 1;
}

def VM_RemF32Op :
    VM_TotalBinaryArithmeticOp<F32, "rem.f32", VM_OPC_RemF32,
                               [VM_ExtF32]> {
  let summary = [{floating point remainder operation}];
  let hasFolder = 1;
}

def VM_RemF64Op :
    VM_TotalBinaryArithmeticOp<F64, "rem.f64", VM_OPC_RemF64,
                               [VM_ExtF64]> {
  let summary = [{floating point remainder operation}];
  let hasFolder = 1;
}

def VM_FMAF32Op :
    VM_TotalTernaryArithmeticOp<F32, "fma.f32", VM_OPC_FMAF32, [VM_ExtF32]> {
  let summary = [{floating point fused multiply-add operation (a*b+c)}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_FMAF64Op :
    VM_TotalTernaryArithmeticOp<F64, "fma.f64", VM_OPC_FMAF64, [VM_ExtF64]> {
  let summary = [{floating point fused multiply-add operation (a*b+c)}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_AbsF32Op :
    VM_TotalUnaryArithmeticOp<F32, "abs.f32", VM_OPC_AbsF32,
                              [VM_ExtF32]> {
  let summary = [{floating point absolute-value operation}];
  let hasFolder = 1;
}

def VM_AbsF64Op :
    VM_TotalUnaryArithmeticOp<F64, "abs.f64", VM_OPC_AbsF64,
                              [VM_ExtF64]> {
  let summary = [{floating point absolute-value operation}];
  let hasFolder = 1;
}

def VM_NegF32Op :
    VM_TotalUnaryArithmeticOp<F32, "neg.f32", VM_OPC_NegF32,
                             [VM_ExtF32]> {
  let summary = [{floating point negation operation}];
  let hasFolder = 1;
}

def VM_NegF64Op :
    VM_TotalUnaryArithmeticOp<F64, "neg.f64", VM_OPC_NegF64,
                              [VM_ExtF64]> {
  let summary = [{floating point negation operation}];
  let hasFolder = 1;
}

def VM_CeilF32Op :
    VM_TotalUnaryArithmeticOp<F32, "ceil.f32", VM_OPC_CeilF32,
                              [VM_ExtF32]> {
  let summary = [{floating point ceiling operation}];
  let hasFolder = 1;
}

def VM_CeilF64Op :
    VM_TotalUnaryArithmeticOp<F64, "ceil.f64", VM_OPC_CeilF64,
                              [VM_ExtF64]> {
  let summary = [{floating point ceiling operation}];
  let hasFolder = 1;
}

def VM_FloorF32Op :
    VM_TotalUnaryArithmeticOp<F32, "floor.f32", VM_OPC_FloorF32,
                              [VM_ExtF32]> {
  let summary = [{floating point floor operation}];
  let hasFolder = 1;
}

def VM_FloorF64Op :
    VM_TotalUnaryArithmeticOp<F64, "floor.f64", VM_OPC_FloorF64,
                              [VM_ExtF64]> {
  let summary = [{floating point floor operation}];
  let hasFolder = 1;
}

def VM_RoundF32Op :
    VM_TotalUnaryArithmeticOp<F32, "round.f32", VM_OPC_RoundF32,
                              [VM_ExtF32]> {
  let summary = [{rounds the value to the nearest integer away from zero}];
}

def VM_RoundF64Op :
    VM_TotalUnaryArithmeticOp<F64, "round.f64", VM_OPC_RoundF64,
                              [VM_ExtF64]> {
  let summary = [{rounds the value to the nearest integer away from zero}];
}

//===----------------------------------------------------------------------===//
// Floating-point math
//===----------------------------------------------------------------------===//
// These map directly to the `math` dialect.

def VM_AtanF32Op :
    VM_TotalUnaryArithmeticOp<F32, "atan.f32", VM_OPC_AtanF32, [VM_ExtF32]> {
  let summary = [{arcus tangent of the given value}];
}

def VM_AtanF64Op :
    VM_TotalUnaryArithmeticOp<F64, "atan.f64", VM_OPC_AtanF64, [VM_ExtF64]> {
  let summary = [{arcus tangent of the given value}];
}

def VM_Atan2F32Op :
    VM_TotalBinaryArithmeticOp<F32, "atan2.f32", VM_OPC_Atan2F32, [VM_ExtF32]> {
  let summary = [{2-argument arcus tangent of the given values}];
}

def VM_Atan2F64Op :
    VM_TotalBinaryArithmeticOp<F64, "atan2.f64", VM_OPC_Atan2F64, [VM_ExtF64]> {
  let summary = [{2-argument arcus tangent of the given values}];
}

def VM_CosF32Op :
    VM_TotalUnaryArithmeticOp<F32, "cos.f32", VM_OPC_CosF32, [VM_ExtF32]> {
  let summary = [{cosine of the specified value}];
}

def VM_CosF64Op :
    VM_TotalUnaryArithmeticOp<F64, "cos.f64", VM_OPC_CosF64, [VM_ExtF64]> {
  let summary = [{cosine of the specified value}];
}

def VM_SinF32Op :
    VM_TotalUnaryArithmeticOp<F32, "sin.f32", VM_OPC_SinF32, [VM_ExtF32]> {
  let summary = [{sine of the specified value}];
}
def VM_SinF64Op :
    VM_TotalUnaryArithmeticOp<F64, "sin.f64", VM_OPC_SinF64, [VM_ExtF64]> {
  let summary = [{sine of the specified value}];
}

def VM_ExpF32Op :
    VM_TotalUnaryArithmeticOp<F32, "exp.f32", VM_OPC_ExpF32, [VM_ExtF32]> {
  let summary = [{base-e exponential of the specified value}];
}

def VM_ExpF64Op :
    VM_TotalUnaryArithmeticOp<F64, "exp.f64", VM_OPC_ExpF64, [VM_ExtF64]> {
  let summary = [{base-e exponential of the specified value}];
}

def VM_Exp2F32Op :
    VM_TotalUnaryArithmeticOp<F32, "exp2.f32", VM_OPC_Exp2F32, [VM_ExtF32]> {
  let summary = [{base-2 exponential of the specified value}];
}

def VM_Exp2F64Op :
    VM_TotalUnaryArithmeticOp<F64, "exp2.f64", VM_OPC_Exp2F64, [VM_ExtF64]> {
  let summary = [{base-2 exponential of the specified value}];
}

def VM_ExpM1F32Op :
    VM_TotalUnaryArithmeticOp<F32, "expm1.f32", VM_OPC_ExpM1F32, [VM_ExtF32]> {
  let summary = [{base-e exponential of the specified value minus 1}];
}

def VM_ExpM1F64Op :
    VM_TotalUnaryArithmeticOp<F64, "expm1.f64", VM_OPC_ExpM1F64, [VM_ExtF64]> {
  let summary = [{base-e exponential of the specified value minus 1}];
}

def VM_LogF32Op :
    VM_TotalUnaryArithmeticOp<F32, "log.f32", VM_OPC_LogF32, [VM_ExtF32]> {
  let summary = [{base-e logarithm of the specified value}];
}

def VM_LogF64Op :
   VM_TotalUnaryArithmeticOp<F64, "log.f64", VM_OPC_LogF64, [VM_ExtF64]> {
  let summary = [{base-e logarithm of the specified value}];
}

def VM_Log10F32Op :
    VM_TotalUnaryArithmeticOp<F32, "log10.f32", VM_OPC_Log10F32, [VM_ExtF32]> {
  let summary = [{base-10 logarithm of the specified value}];
}
def VM_Log10F64Op :
    VM_TotalUnaryArithmeticOp<F64, "log10.f64", VM_OPC_Log10F64, [VM_ExtF64]> {
  let summary = [{base-10 logarithm of the specified value}];
}

def VM_Log1pF32Op :
    VM_TotalUnaryArithmeticOp<F32, "log1p.f32", VM_OPC_Log1pF32, [VM_ExtF32]> {
  let summary = [{natural logarithm of one plus the given value}];
}

def VM_Log1pF64Op :
    VM_TotalUnaryArithmeticOp<F64, "log1p.f64", VM_OPC_Log1pF64, [VM_ExtF64]> {
  let summary = [{natural logarithm of one plus the given value}];
}

def VM_Log2F32Op :
    VM_TotalUnaryArithmeticOp<F32, "log2.f32", VM_OPC_Log2F32, [VM_ExtF32]> {
  let summary = [{base-2 logarithm of the specified value}];
}

def VM_Log2F64Op :
    VM_TotalUnaryArithmeticOp<F64, "log2.f64", VM_OPC_Log2F64, [VM_ExtF64]> {
  let summary = [{base-2 logarithm of the specified value}];
}

def VM_PowF32Op :
    VM_TotalBinaryArithmeticOp<F32, "pow.f32", VM_OPC_PowF32, [VM_ExtF32]> {
  let summary = [{floating point raised to the power of operation}];
}

def VM_PowF64Op :
    VM_TotalBinaryArithmeticOp<F64, "pow.f64", VM_OPC_PowF64, [VM_ExtF64]> {
  let summary = [{floating point raised to the power of operation}];
}

def VM_RsqrtF32Op :
    VM_TotalUnaryArithmeticOp<F32, "rsqrt.f32", VM_OPC_RsqrtF32, [VM_ExtF32]> {
  let summary = [{reciprocal of sqrt (1 / sqrt of the specified value)}];
}

def VM_RsqrtF64Op :
    VM_TotalUnaryArithmeticOp<F64, "rsqrt.f64", VM_OPC_RsqrtF64, [VM_ExtF64]> {
  let summary = [{reciprocal of sqrt (1 / sqrt of the specified value)}];
}

def VM_SqrtF32Op :
    VM_TotalUnaryArithmeticOp<F32, "sqrt.f32", VM_OPC_SqrtF32, [VM_ExtF32]> {
  let summary = [{sqrt of the specified value}];
  let hasFolder = 1;
}

def VM_SqrtF64Op :
    VM_TotalUnaryArithmeticOp<F64, "sqrt.f64", VM_OPC_SqrtF64, [VM_ExtF64]> {
  let summary = [{sqrt of the specified value}];
  let hasFolder = 1;
}

def VM_TanhF32Op :
    VM_TotalUnaryArithmeticOp<F32, "tanh.f32", VM_OPC_TanhF32, [VM_ExtF32]> {
  let summary = [{hyperbolic tangent of the specified value}];
}

def VM_TanhF64Op :
    VM_TotalUnaryArithmeticOp<F64, "tanh.f64", VM_OPC_TanhF64, [VM_ExtF64]> {
  let summary = [{hyperbolic tangent of the specified value}];
}

def VM_ErfF32Op :
    VM_TotalUnaryArithmeticOp<F32, "erf.f32", VM_OPC_ErfF32, [VM_ExtF32]> {
  let summary = [{computes the error function of the specified value}];
}

def VM_ErfF64Op :
    VM_TotalUnaryArithmeticOp<F64, "erf.f64", VM_OPC_ErfF64, [VM_ExtF64]> {
  let summary = [{computes the error function of the specified value}];
}

//===----------------------------------------------------------------------===//
// Integer bit manipulation
//===----------------------------------------------------------------------===//

def VM_NotI32Op :
    VM_TotalUnaryArithmeticOp<I32, "not.i32", VM_OPC_NotI32> {
  let summary = [{integer binary not operation}];
  let hasFolder = 1;
}

def VM_NotI64Op :
    VM_TotalUnaryArithmeticOp<I64, "not.i64", VM_OPC_NotI64> {
  let summary = [{integer binary not operation}];
  let hasFolder = 1;
}

def VM_AndI32Op :
    VM_TotalBinaryArithmeticOp<I32, "and.i32", VM_OPC_AndI32, [Commutative]> {
  let summary = [{integer binary and operation}];
  let hasFolder = 1;
}

def VM_AndI64Op :
    VM_TotalBinaryArithmeticOp<I64, "and.i64", VM_OPC_AndI64, [Commutative]> {
  let summary = [{integer binary and operation}];
  let hasFolder = 1;
}

def VM_OrI32Op :
    VM_TotalBinaryArithmeticOp<I32, "or.i32", VM_OPC_OrI32, [Commutative]> {
  let summary = [{integer binary or operation}];
  let hasFolder = 1;
}

def VM_OrI64Op :
    VM_TotalBinaryArithmeticOp<I64, "or.i64", VM_OPC_OrI64, [Commutative]> {
  let summary = [{integer binary or operation}];
  let hasFolder = 1;
}

def VM_XorI32Op :
    VM_TotalBinaryArithmeticOp<I32, "xor.i32", VM_OPC_XorI32, [Commutative]> {
  let summary = [{integer binary exclusive-or operation}];
  let hasFolder = 1;
}

def VM_XorI64Op :
    VM_TotalBinaryArithmeticOp<I64, "xor.i64", VM_OPC_XorI64, [Commutative]> {
  let summary = [{integer binary exclusive-or operation}];
  let hasFolder = 1;
}

def VM_CtlzI32Op :
    VM_TotalUnaryArithmeticOp<I32, "ctlz.i32", VM_OPC_CtlzI32> {
  let summary = [{counts the leading zeros in an integer value}];
  let hasFolder = 1;
}

def VM_CtlzI64Op :
    VM_TotalUnaryArithmeticOp<I64, "ctlz.i64", VM_OPC_CtlzI64> {
  let summary = [{counts the leading zeros in an integer value}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Bitwise shifts and rotates
//===----------------------------------------------------------------------===//

class VM_ShiftArithmeticOp<I type, string mnemonic, VM_OPC opcode,
                           list<Trait> traits = []> :
    VM_TrivialOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["operand", "result"]>,
    ])> {
  let description = [{
    Shifts the operand in a direction by the number of bits specified.
  }];

  let arguments = (ins
    type:$operand,
    I32:$amount
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = [{
    $operand `,` $amount attr-dict `:` type($operand)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncOperand<"amount", 1>,
    VM_EncResult<"result">,
  ];
}

def VM_ShlI32Op : VM_ShiftArithmeticOp<I32, "shl.i32", VM_OPC_ShlI32> {
  let summary = [{integer shift left operation}];
  let hasFolder = 1;
}

def VM_ShlI64Op : VM_ShiftArithmeticOp<I64, "shl.i64", VM_OPC_ShlI64> {
  let summary = [{integer shift left operation}];
  let hasFolder = 1;
}

def VM_ShrI32SOp : VM_ShiftArithmeticOp<I32, "shr.i32.s", VM_OPC_ShrI32S> {
  let summary = [{signed integer (arithmetic) shift right operation}];
  let hasFolder = 1;
}

def VM_ShrI64SOp : VM_ShiftArithmeticOp<I64, "shr.i64.s", VM_OPC_ShrI64S> {
  let summary = [{signed integer (arithmetic) shift right operation}];
  let hasFolder = 1;
}

def VM_ShrI32UOp : VM_ShiftArithmeticOp<I32, "shr.i32.u", VM_OPC_ShrI32U> {
  let summary = [{unsigned integer (logical) shift right operation}];
  let hasFolder = 1;
}

def VM_ShrI64UOp : VM_ShiftArithmeticOp<I64, "shr.i64.u", VM_OPC_ShrI64U> {
  let summary = [{unsigned integer (logical) shift right operation}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===----------------------------------------------------------------------===//

class VM_ConversionOp<Type src_type, Type dst_type, string mnemonic,
                      VM_OPC opcode, list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let arguments = (ins
    src_type:$operand
  );
  let results = (outs
    dst_type:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncResult<"result">,
  ];
}

class VM_ConversionPseudoOp<Type src_type, Type dst_type, string mnemonic,
                            list<Trait> traits = []> :
    VM_PureOp<mnemonic, traits> {
  let arguments = (ins
    src_type:$operand
  );
  let results = (outs
    dst_type:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` type($result)
  }];

  let hasCanonicalizer = 1;
}

def VM_TruncI32I8Op :
    VM_ConversionOp<I32, I32, "trunc.i32.i8", VM_OPC_TruncI32I8> {
  let summary = [{integer truncate to 8 bits}];
  let hasFolder = 1;
}

def VM_TruncI32I16Op :
    VM_ConversionOp<I32, I32, "trunc.i32.i16", VM_OPC_TruncI32I16> {
  let summary = [{integer truncate to 16 bits}];
  let hasFolder = 1;
}

def VM_TruncI64I8Op :
    VM_ConversionPseudoOp<I64, I32, "trunc.i64.i8"> {
  let summary = [{integer truncate to 8 bits}];
  let hasFolder = 1;
}

def VM_TruncI64I16Op :
    VM_ConversionPseudoOp<I64, I32, "trunc.i64.i16"> {
  let summary = [{integer truncate to 16 bits}];
  let hasFolder = 1;
}

def VM_TruncI64I32Op :
    VM_ConversionOp<I64, I32, "trunc.i64.i32", VM_OPC_TruncI64I32> {
  let summary = [{integer truncate to 32 bits}];
  let hasFolder = 1;
}

def VM_TruncF64F32Op :
    VM_ConversionOp<F64, F32, "trunc.f64.f32", VM_OPC_TruncF64F32,
                    [VM_ExtF64]> {
  let summary = [{floating-point truncate to 32 bits}];
  let hasFolder = 1;
}

def VM_ExtI8I32SOp :
    VM_ConversionOp<I32, I32, "ext.i8.i32.s", VM_OPC_ExtI8I32S> {
  let summary = [{integer sign extend 8 bits to 32 bits}];
  let hasFolder = 1;
}

def VM_ExtI8I32UOp :
    VM_ConversionOp<I32, I32, "ext.i8.i32.u", VM_OPC_ExtI8I32U> {
  let summary = [{integer zero extend 8 bits to 32 bits}];
  let hasFolder = 1;
}

def VM_ExtI16I32SOp :
    VM_ConversionOp<I32, I32, "ext.i16.i32.s", VM_OPC_ExtI16I32S> {
  let summary = [{integer sign extend 16 bits to 32 bits}];
  let hasFolder = 1;
}

def VM_ExtI16I32UOp :
    VM_ConversionOp<I32, I32, "ext.i16.i32.u", VM_OPC_ExtI16I32U> {
  let summary = [{integer zero extend 16 bits to 32 bits}];
  let hasFolder = 1;
}

def VM_ExtI8I64SOp :
    VM_ConversionPseudoOp<I32, I64, "ext.i8.i64.s"> {
  let summary = [{integer sign extend 8 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_ExtI8I64UOp :
    VM_ConversionPseudoOp<I32, I64, "ext.i8.i64.u"> {
  let summary = [{integer zero extend 8 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_ExtI16I64SOp :
    VM_ConversionPseudoOp<I32, I64, "ext.i16.i64.s"> {
  let summary = [{integer sign extend 16 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_ExtI16I64UOp :
    VM_ConversionPseudoOp<I32, I64, "ext.i16.i64.u"> {
  let summary = [{integer zero extend 16 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_ExtI32I64SOp :
    VM_ConversionOp<I32, I64, "ext.i32.i64.s", VM_OPC_ExtI32I64S> {
  let summary = [{integer sign extend 32 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_ExtI32I64UOp :
    VM_ConversionOp<I32, I64, "ext.i32.i64.u", VM_OPC_ExtI32I64U> {
  let summary = [{integer zero extend 32 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_ExtF32F64Op :
    VM_ConversionOp<F32, F64, "ext.f32.f64", VM_OPC_ExtF32F64,
                    [VM_ExtF64]> {
  let summary = [{floating-point zero extend 32 bits to 64 bits}];
  let hasFolder = 1;
}

def VM_CastSI32F32Op :
    VM_ConversionOp<I32, F32, "cast.si32.f32", VM_OPC_CastSI32F32,
                    [VM_ExtF32]> {
  let summary = [{cast from a signed integer to a float-point value}];
  let hasFolder = 1;
}

def VM_CastUI32F32Op :
    VM_ConversionOp<I32, F32, "cast.ui32.f32", VM_OPC_CastUI32F32,
                    [VM_ExtF32]> {
  let summary = [{cast from an unsigned integer to a float-point value}];
  let hasFolder = 1;
}

def VM_CastF32SI32Op :
    VM_ConversionOp<F32, I32, "cast.f32.si32", VM_OPC_CastF32SI32,
                    [VM_ExtF32]> {
  let summary = [{cast from a float-point value to a signed integer}];
  let hasFolder = 1;
}

def VM_CastF32UI32Op :
    VM_ConversionOp<F32, I32, "cast.f32.ui32", VM_OPC_CastF32UI32,
                    [VM_ExtF32]> {
  let summary = [{cast from an float-point value to an unsigned integer}];
  let hasFolder = 1;
}

def VM_BitcastI32F32Op :
    VM_ConversionOp<I32, F32, "bitcast.i32.f32", VM_OPC_BitcastI32F32,
                    [VM_ExtF32]> {
  let summary = [{bitcast from a 32-bit integer to a 32-bit float-point value}];
}

def VM_BitcastF32I32Op :
    VM_ConversionOp<F32, I32, "bitcast.f32.i32", VM_OPC_BitcastF32I32,
                    [VM_ExtF32]> {
  let summary = [{bitcast from a 32-bit float-point value to a 32-bit integer}];
}

def VM_BitcastI64F64Op :
    VM_ConversionOp<I64, F64, "bitcast.i64.f64", VM_OPC_BitcastI64F64,
                    [VM_ExtF64]> {
  let summary = [{bitcast from a 64-bit integer to a 64-bit float-point value}];
}

def VM_BitcastF64I64Op :
    VM_ConversionOp<F64, I64, "bitcast.f64.i64", VM_OPC_BitcastF64I64,
                    [VM_ExtF64]> {
  let summary = [{bitcast from a 64-bit float-point value to a 64-bit integer}];
}

//===----------------------------------------------------------------------===//
// Native reduction (horizontal) arithmetic
//===----------------------------------------------------------------------===//

// TODO(benvanik): used for stride/element count/etc.

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

class VM_UnaryComparisonOp<Type type, string mnemonic, VM_OPC opcode,
                           list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let description = [{
    Compares a single operand against the specified predicate.
  }];

  let arguments = (ins
    type:$operand
  );
  let results = (outs
    VM_CondValue:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncResult<"result">,
  ];
}

class VM_UnaryComparisonPseudoOp<Type type, string mnemonic,
                                 list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      VM_PseudoOp,
    ])> {
  let description = [{
    Compares an operand with the specified predicate.
  }];

  let arguments = (ins
    type:$operand
  );
  let results = (outs
    VM_CondValue:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($operand)
  }];
}

class VM_BinaryComparisonOp<Type type, string mnemonic, VM_OPC opcode,
                            list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["lhs", "rhs"]>,
    ])> {
  let description = [{
    Compares two operands with the specified predicate.
  }];

  let arguments = (ins
    type:$lhs,
    type:$rhs
  );
  let results = (outs
    VM_CondValue:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($lhs)
  }];

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"lhs", 0>,
    VM_EncOperand<"rhs", 1>,
    VM_EncResult<"result">,
  ];
}

class VM_BinaryComparisonPseudoOp<Type type, string mnemonic,
                                  list<Trait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      AllTypesMatch<["lhs", "rhs"]>,
      VM_PseudoOp,
    ])> {
  let description = [{
    Compares two operands with the specified predicate.
  }];

  let arguments = (ins
    type:$lhs,
    type:$rhs
  );
  let results = (outs
    VM_CondValue:$result
  );

  let assemblyFormat = [{
    operands attr-dict `:` type($lhs)
  }];
}

def VM_CmpEQI32Op :
    VM_BinaryComparisonOp<I32, "cmp.eq.i32", VM_OPC_CmpEQI32, [
        Commutative,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{integer equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpEQI64Op :
    VM_BinaryComparisonOp<I64, "cmp.eq.i64", VM_OPC_CmpEQI64, [
        Commutative,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{integer equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEI32Op :
    VM_BinaryComparisonOp<I32, "cmp.ne.i32", VM_OPC_CmpNEI32, [
        Commutative,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{integer inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEI64Op :
    VM_BinaryComparisonOp<I64, "cmp.ne.i64", VM_OPC_CmpNEI64, [
        Commutative,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{integer inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTI32SOp :
    VM_BinaryComparisonOp<I32, "cmp.lt.i32.s", VM_OPC_CmpLTI32S, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTI64SOp :
    VM_BinaryComparisonOp<I64, "cmp.lt.i64.s", VM_OPC_CmpLTI64S, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTI32UOp :
    VM_BinaryComparisonOp<I32, "cmp.lt.i32.u", VM_OPC_CmpLTI32U, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTI64UOp :
    VM_BinaryComparisonOp<I64, "cmp.lt.i64.u", VM_OPC_CmpLTI64U, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEI32SOp :
    VM_BinaryComparisonPseudoOp<I32, "cmp.lte.i32.s", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer less-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEI64SOp :
    VM_BinaryComparisonPseudoOp<I64, "cmp.lte.i64.s", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer less-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEI32UOp :
    VM_BinaryComparisonPseudoOp<I32, "cmp.lte.i32.u", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer less-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEI64UOp :
    VM_BinaryComparisonPseudoOp<I64, "cmp.lte.i64.u", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer less-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTI32SOp :
    VM_BinaryComparisonPseudoOp<I32, "cmp.gt.i32.s", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTI64SOp :
    VM_BinaryComparisonPseudoOp<I64, "cmp.gt.i64.s", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTI32UOp :
    VM_BinaryComparisonPseudoOp<I32, "cmp.gt.i32.u", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTI64UOp :
    VM_BinaryComparisonPseudoOp<I64, "cmp.gt.i64.u", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEI32SOp :
    VM_BinaryComparisonPseudoOp<I32, "cmp.gte.i32.s", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEI64SOp :
    VM_BinaryComparisonPseudoOp<I64, "cmp.gte.i64.s", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{signed integer greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEI32UOp :
    VM_BinaryComparisonPseudoOp<I32, "cmp.gte.i32.u", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEI64UOp :
    VM_BinaryComparisonPseudoOp<I64, "cmp.gte.i64.u", [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{unsigned integer greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNZI32Op :
    VM_UnaryComparisonOp<I32, "cmp.nz.i32", VM_OPC_CmpNZI32, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{integer non-zero comparison operation}];
  let description = [{
    Compares the given integer operand for a non-zero value.
  }];
  let hasFolder = 1;
}

def VM_CmpNZI64Op :
    VM_UnaryComparisonOp<I64, "cmp.nz.i64", VM_OPC_CmpNZI64, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{integer non-zero comparison operation}];
  let description = [{
    Compares the given integer operand for a non-zero value.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Floating-point comparison
//===----------------------------------------------------------------------===//

def VM_CmpEQF32OOp :
    VM_BinaryComparisonOp<F32, "cmp.eq.f32.o", VM_OPC_CmpEQF32O,
                          [VM_ExtF32, Commutative]> {
  let summary = [{ordered floating-point equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpEQF64OOp :
    VM_BinaryComparisonOp<F64, "cmp.eq.f64.o", VM_OPC_CmpEQF64O,
                          [VM_ExtF64, Commutative]> {
  let summary = [{ordered floating-point equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpEQF32UOp :
    VM_BinaryComparisonOp<F32, "cmp.eq.f32.u", VM_OPC_CmpEQF32U,
                          [VM_ExtF32, Commutative]> {
  let summary = [{unordered floating-point equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpEQF64UOp :
    VM_BinaryComparisonOp<F64, "cmp.eq.f64.u", VM_OPC_CmpEQF64U,
                          [VM_ExtF64, Commutative]> {
  let summary = [{unordered floating-point equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpEQF32NearOp :
    VM_BinaryComparisonPseudoOp<F32, "cmp.eq.f32.near",
                          [VM_ExtF32, Commutative]> {
  let summary = [{near floating-point equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpEQF64NearOp :
    VM_BinaryComparisonPseudoOp<F64, "cmp.eq.f64.near",
                          [VM_ExtF64, Commutative]> {
  let summary = [{near floating-point equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEF32OOp :
    VM_BinaryComparisonOp<F32, "cmp.ne.f32.o", VM_OPC_CmpNEF32O,
                          [VM_ExtF32, Commutative]> {
  let summary = [{ordered floating-point inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEF64OOp :
    VM_BinaryComparisonOp<F64, "cmp.ne.f64.o", VM_OPC_CmpNEF64O,
                          [VM_ExtF64, Commutative]> {
  let summary = [{ordered floating-point inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEF32UOp :
    VM_BinaryComparisonOp<F32, "cmp.ne.f32.u", VM_OPC_CmpNEF32U,
                          [VM_ExtF32, Commutative]> {
  let summary = [{unordered floating-point inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEF64UOp :
    VM_BinaryComparisonOp<F64, "cmp.ne.f64.u", VM_OPC_CmpNEF64U,
                          [VM_ExtF64, Commutative]> {
  let summary = [{unordered floating-point inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTF32OOp :
    VM_BinaryComparisonOp<F32, "cmp.lt.f32.o", VM_OPC_CmpLTF32O,
                          [VM_ExtF32]> {
  let summary = [{ordered floating-point less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTF64OOp :
    VM_BinaryComparisonOp<F64, "cmp.lt.f64.o", VM_OPC_CmpLTF64O,
                          [VM_ExtF64]> {
  let summary = [{ordered floating-point less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTF32UOp :
    VM_BinaryComparisonOp<F32, "cmp.lt.f32.u", VM_OPC_CmpLTF32U,
                          [VM_ExtF32]> {
  let summary = [{unordered floating-point less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTF64UOp :
    VM_BinaryComparisonOp<F64, "cmp.lt.f64.u", VM_OPC_CmpLTF64U,
                          [VM_ExtF64]> {
  let summary = [{unordered floating-point less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEF32OOp :
    VM_BinaryComparisonOp<F32, "cmp.lte.f32.o", VM_OPC_CmpLTEF32O,
                                [VM_ExtF32]> {
  let summary = [{ordered floating-point less-than-or-equal comparison operation}];
  let hasFolder = 1;
}

def VM_CmpLTEF64OOp :
    VM_BinaryComparisonOp<F64, "cmp.lte.f64.o", VM_OPC_CmpLTEF64O,
                          [VM_ExtF64]> {
  let summary = [{ordered floating-point less-than-or-equal comparison operation}];
  let hasFolder = 1;
}

def VM_CmpLTEF32UOp :
    VM_BinaryComparisonOp<F32, "cmp.lte.f32.u", VM_OPC_CmpLTEF32U,
                          [VM_ExtF32]> {
  let summary = [{unordered floating-point less-than-or-equal comparison operation}];
  let hasFolder = 1;
}

def VM_CmpLTEF64UOp :
    VM_BinaryComparisonOp<F64, "cmp.lte.f64.u", VM_OPC_CmpLTEF64U,
                          [VM_ExtF64]> {
  let summary = [{unordered floating-point less-than-or-equal comparison operation}];
  let hasFolder = 1;
}

def VM_CmpGTF32OOp :
    VM_BinaryComparisonPseudoOp<F32, "cmp.gt.f32.o",
                                [VM_ExtF32]> {
  let summary = [{ordered floating-point greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTF64OOp :
    VM_BinaryComparisonPseudoOp<F64, "cmp.gt.f64.o",
                                [VM_ExtF64]> {
  let summary = [{ordered floating-point greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTF32UOp :
    VM_BinaryComparisonPseudoOp<F32, "cmp.gt.f32.u",
                                [VM_ExtF32]> {
  let summary = [{unordered floating-point greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTF64UOp :
    VM_BinaryComparisonPseudoOp<F64, "cmp.gt.f64.u",
                                [VM_ExtF64]> {
  let summary = [{unordered floating-point greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEF32OOp :
    VM_BinaryComparisonPseudoOp<F32, "cmp.gte.f32.o",
                                [VM_ExtF32]> {
  let summary = [{ordered floating-point greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEF64OOp :
    VM_BinaryComparisonPseudoOp<F64, "cmp.gte.f64.o",
                                [VM_ExtF64]> {
  let summary = [{ordered floating-point greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEF32UOp :
    VM_BinaryComparisonPseudoOp<F32, "cmp.gte.f32.u",
                                [VM_ExtF32]> {
  let summary = [{unordered floating-point greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEF64UOp :
    VM_BinaryComparisonPseudoOp<F64, "cmp.gte.f64.u",
                                [VM_ExtF64]> {
  let summary = [{unordered floating-point greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNZF32OOp :
    VM_UnaryComparisonPseudoOp<F32, "cmp.nz.f32.o",
                               [VM_ExtF32]> {
  let summary = [{ordered floating-point non-zero comparison operation}];
  let description = [{
    Compares the given floating-point operand for a non-zero value.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNZF64OOp :
    VM_UnaryComparisonPseudoOp<F64, "cmp.nz.f64.o",
                               [VM_ExtF64]> {
  let summary = [{ordered floating-point non-zero comparison operation}];
  let description = [{
    Compares the given floating-point operand for a non-zero value.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNZF32UOp :
    VM_UnaryComparisonPseudoOp<F32, "cmp.nz.f32.u",
                               [VM_ExtF32]> {
  let summary = [{unordered floating-point non-zero comparison operation}];
  let description = [{
    Compares the given floating-point operand for a non-zero value.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNZF64UOp :
    VM_UnaryComparisonPseudoOp<F64, "cmp.nz.f64.u",
                               [VM_ExtF64]> {
  let summary = [{unordered floating-point non-zero comparison operation}];
  let description = [{
    Compares the given floating-point operand for a non-zero value.
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNaNF32Op :
    VM_UnaryComparisonOp<F32, "cmp.nan.f32", VM_OPC_CmpNaNF32,
                         [VM_ExtF32]> {
  let summary = [{floating-point NaN comparison operation}];
  let description = [{
    Returns 1 if the value is NaN.
  }];
  let hasFolder = 1;
}

def VM_CmpNaNF64Op :
    VM_UnaryComparisonOp<F64, "cmp.nan.f64", VM_OPC_CmpNaNF64,
                         [VM_ExtF64]> {
  let summary = [{floating-point NaN comparison operation}];
  let description = [{
    Returns 1 if the value is NaN.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// vm.ref comparison
//===----------------------------------------------------------------------===//

def VM_CmpEQRefOp :
    VM_BinaryComparisonOp<VM_AnyRef, "cmp.eq.ref", VM_OPC_CmpEQRef, [
        Commutative,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{ref<T> equality comparison operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// TODO(benvanik): remove and just rely on eq & flipped cond_br?
def VM_CmpNERefOp :
    VM_BinaryComparisonOp<VM_AnyRef, "cmp.ne.ref", VM_OPC_CmpNERef, [
        Commutative,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{ref<T> inequality comparison operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_CmpNZRefOp :
    VM_UnaryComparisonOp<VM_AnyRef, "cmp.nz.ref", VM_OPC_CmpNZRef, [
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = [{ref<T> non-zero comparison operation}];
  let description = [{
    Compares the given ref operand for a non-zero/null value.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Control flow
//===----------------------------------------------------------------------===//

def VM_BranchOp : VM_Op<"br", [
    DeclareOpInterfaceMethods<BranchOpInterface>,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    Terminator,
  ]> {
  let summary = [{unconditional branch operation}];
  let description = [{
    Represents an unconditional branch operation that branches to a target block
    with the given set of arguments.

    ```
    ^bb0(...):
      vm.br ^bb1(%a)
    ^bb1(%blockArg1):
      ...
   ```
  }];

  let arguments = (ins
    Variadic<VM_AnyType>:$destOperands
  );

  let successors = (successor
    AnySuccessor:$dest
  );

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Branch>,
    VM_EncBranch<"dest", "getOperands", 0>,
  ];

  let builders = [
    OpBuilder<(ins "Block *":$dest, CArg<"ValueRange", "{}">:$destOperands),
    [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>
  ];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizer = 1;
}

def VM_CondBranchOp : VM_Op<"cond_br", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<BranchOpInterface>,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    Terminator,
  ]> {
  let summary = [{conditional branch operation}];
  let description = [{
    Represents a conditional branch operation that branches to one of the two
    target blocks with the given set of arguments.

    ```
    ^bb0(...):
      vm.cond_br %condition, ^bb1(%a), ^bb2(%b)
    ^bb1(%blockArg1):
      ...
    ^bb2(%blockArg2):
      ...
   ```
  }];

  let arguments = (ins
    VM_CondValue:$condition,
    Variadic<VM_AnyType>:$trueDestOperands,
    Variadic<VM_AnyType>:$falseDestOperands
  );

  let successors = (successor
    AnySuccessor:$trueDest,
    AnySuccessor:$falseDest
  );

  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_CondBranch>,
    VM_EncOperand<"condition", 0>,
    VM_EncBranch<"trueDest", "getTrueOperands", 0>,
    VM_EncBranch<"falseDest", "getFalseOperands", 1>,
  ];

  let builders = [
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "ValueRange":$trueOperands, "Block *":$falseDest,
      "ValueRange":$falseOperands),
    [{
      build($_builder, $_state, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    }]>,
  OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
    "Block *":$falseDest, CArg<"ValueRange", "{}">:$falseOperands),
  [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    }]>
  ];

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    /// Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return getTrueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getTrueDestOperandsMutable().erase(index);
    }

    /// Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getFalseOperands() { return getFalseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getFalseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let hasCanonicalizer = 1;
}

class VM_CallBaseOp<string mnemonic, list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
      CallOpInterface,
    ])> {
  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getOperation()->getAttrOfType<FlatSymbolRefAttr>("callee");
    }
  }];
}

def VM_CallOp : VM_CallBaseOp<"call"> {
  let summary = [{call operation}];
  let description = [{
    Calls an internal VM function with the given arguments.
  }];

  let arguments = (ins
    VM_FuncRefAttr:$callee,
    Variadic<VM_AnyType>:$operands
  );
  let results = (outs
    Variadic<VM_AnyType>:$results
  );

  let assemblyFormat = [{
    $callee `(` operands `)` attr-dict `:` functional-type(operands, results)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Call>,
    VM_EncFuncAttr<"callee">,
    VM_EncVariadicOperands<"operands">,
    VM_EncVariadicResults<"results">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "IREE::VM::FuncOp":$callee,
      CArg<"ValueRange", "{}">:$operands),
    [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", mlir::SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "FlatSymbolRefAttr":$callee,
      "mlir::TypeRange":$resultTypes, CArg<"ValueRange", "{}">:$operands),
    [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(resultTypes);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "mlir::TypeRange":$resultTypes,
      CArg<"ValueRange", "{}">:$operands),
    [{
      build($_builder, $_state,
            mlir::SymbolRefAttr::get($_builder.getContext(), callee),
            resultTypes, operands);
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def VM_CallVariadicOp : VM_CallBaseOp<"call.variadic"> {
  let summary = [{call operation with variadic arguments}];
  let description = [{
    Calls an internal VM function with the given arguments. One or more of the
    arguments may be variadic, encoded as segmented sized operand lists.

    Variadic arguments must be specified with a total count in the segment_sizes
    attribute.
  }];

  let arguments = (ins
    VM_FuncRefAttr:$callee,
    SignlessIntElementsAttr<16>:$segment_sizes,
    TypeArrayAttr:$segment_types,
    Variadic<VM_AnyType>:$operands
  );
  let results = (outs
    Variadic<VM_AnyType>:$results
  );

  let encoding = [
    VM_EncOpcode<VM_OPC_CallVariadic>,
    VM_EncFuncAttr<"callee">,
    VM_EncPrimitiveArrayAttr<"segment_sizes", 16>,
    VM_EncVariadicOperands<"operands">,
    VM_EncVariadicResults<"results">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$callee,
      "ArrayRef<Type>":$resultTypes,
      CArg<"ArrayRef<int16_t>", "{}">:$segmentSizes,
      CArg<"ArrayRef<Type>", "{}">:$segmentTypes,
      CArg<"ValueRange", "{}">:$operands),
    [{
      $_state.addAttribute("callee", callee);
      $_state.addAttribute("segment_sizes",
          DenseIntElementsAttr::get(
              VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                              $_builder.getIntegerType(16)),
              segmentSizes));
      $_state.addAttribute("segment_types", $_builder.getArrayAttr(
          llvm::to_vector<4>(llvm::map_range(segmentTypes, [&](Type type) {
              return TypeAttr::get(type).cast<Attribute>();
          }))));
      $_state.addOperands(operands);
      $_state.addTypes(resultTypes);
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def VM_ReturnOp : VM_Op<"return", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    Pure,
    ReturnLike,
    Terminator,
  ]> {
  let summary = "return operation";
  let description = [{
    Represents a return operation within a function.

    ```
    vm.func @foo(%0: i32, %1: f8) -> (i32, f8) {
      vm.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins
    Variadic<VM_AnyType>:$operands
  );

  let assemblyFormat = [{
    attr-dict ($operands^ `:` type($operands))?
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Return>,
    VM_EncVariadicOperands<"operands">,
  ];

  let builders = [
    OpBuilder<(ins),
    [{
      build($_builder, $_state, std::nullopt);
    }]>,
  ];
}

def VM_FailOp : VM_Op<"fail", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    Terminator,
  ]> {
  let summary = [{raises a global failure}];
  let description = [{
    Signals a runtime failure that causes the entire active invocation - and
    possibly *all* in-flight and pending invocations - to fail with the given
    status. The status will be propagated back via the available runtime error
    handling mechanisms such as semaphores or synchronous invocation results.

    As the IREE execution model is deeply pipelined it's possible that failures
    have a latency between when they are emitted and when the application can
    observe the failure. It's also possible that other work that is in-flight
    or pending when the failure occurs will complete.

    ```
    %statusCode = vm.const.i32 9
    vm.fail %statusCode, "oh no!"
    ```
  }];

  let arguments = (ins
    Util_Status:$status,
    OptionalAttr<StrAttr>:$message
  );

  let assemblyFormat = [{
    $status (`,` $message^)? attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Fail>,
    VM_EncOperand<"status", 0>,
    VM_EncStrAttr<"message">,
  ];

  let builders = [
    OpBuilder<(ins "Value":$status, CArg<"StringRef", [{""}]>:$message),
    [{
      build(
          $_builder, $_state, status,
          message.empty() ? StringAttr{} : $_builder.getStringAttr(message));
    }]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult verify() {
      return verifyFailOp(getOperation(), getStatus());
    }
  }];
}

def VM_CondFailOp : VM_Op<"cond_fail", [
    VM_PseudoOp,
  ]> {
  let summary = [{raises a global failure if the condition is true}];
  let description = [{
    When the condition is true this signals a runtime failure that causes the
    entire active invocation - and possibly *all* in-flight and pending
    invocations - to fail with the given status. The status will be propagated
    back via the available runtime error handling mechanisms such as semaphores
    or synchronous invocation results.

    As the IREE execution model is deeply pipelined it's possible that failures
    have a latency between when they are emitted and when the application can
    observe the failure. It's also possible that other work that is in-flight
    or pending when the failure occurs will complete.

    This is implemented as a pseudo-op that transforms into a vm.fail operation
    guarded by the condition.

    ```
    %nz = vm.cmp.nz.i32 %value : i32
    %statusCode = vm.const.i32 9
    vm.cond_fail %nz, %statusCode, "expected non-zero"
    ```
  }];

  let arguments = (ins
    VM_CondValue:$condition,
    Util_Status:$status,
    OptionalAttr<StrAttr>:$message
  );

  let builders = [
    OpBuilder<(ins "Value":$condition, "Value":$status,
      CArg<"StringRef", [{""}]>:$message),
    [{
      build(
          $_builder, $_state, condition, status,
          message.empty() ? StringAttr{} : $_builder.getStringAttr(message));
    }]>,
    OpBuilder<(ins "Value":$status, CArg<"StringRef", [{""}]>:$message),
    [{
      build($_builder, $_state, status, status, message);
    }]>,
  ];

  let extraClassDeclaration = [{
    LogicalResult verify() {
      return verifyFailOp(getOperation(), getStatus());
    }
  }];

  let hasCanonicalizer = 1;
}

class VM_CheckOp<string mnemonic, list<Trait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      VM_PseudoOp,
    ])> {
  let summary = [{raises a global failure if the condition is true}];
  let description = [{
    When the condition is true this signals a runtime failure that causes the
    entire active invocation - and possibly *all* in-flight and pending
    invocations - to fail. The status will be propagated back via the available
    runtime error handling mechanisms such as semaphores or synchronous
    invocation results.

    This is implemented as a pseudo-op that transforms into a vm.cond_fail
    operation.

    ```
    vm.check.eq %a, %b, "a == b" : i32
    vm.check.nz %ref, "!null" : !vm.ref<?>
    ```
  }];
}

class VM_UnaryCheckOp<string mnemonic, list<Trait> traits = []> :
    VM_CheckOp<mnemonic, traits> {
  let arguments = (ins
    VM_AnyType:$value,
    OptionalAttr<StrAttr>:$message
  );


  let assemblyFormat = [{
    $value (`,` $message^)? attr-dict `:` type($value)
  }];

  let builders = [
    OpBuilder<(ins "Value":$value, CArg<"StringRef", [{""}]>:$message),
    [{
      build(
          $_builder, $_state, value,
          message.empty() ? StringAttr{} : $_builder.getStringAttr(message));
    }]>,
  ];
}

class VM_BinaryCheckOp<string mnemonic, list<Trait> traits = []> :
    VM_CheckOp<mnemonic, !listconcat(traits, [
      AllTypesMatch<["lhs", "rhs"]>,
    ])> {
  let arguments = (ins
    VM_AnyType:$lhs,
    VM_AnyType:$rhs,
    OptionalAttr<StrAttr>:$message
  );

  let assemblyFormat = [{
    $lhs `,` $rhs (`,` $message^)? attr-dict `:` type($lhs)
  }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      CArg<"StringRef", [{""}]>:$message),
    [{
      build(
          $_builder, $_state, lhs, rhs,
          message.empty() ? StringAttr{} : $_builder.getStringAttr(message));
    }]>,
  ];
}

def VM_CheckEQOp : VM_BinaryCheckOp<"check.eq", [Commutative]> {
  let hasCanonicalizer = 1;
}

def VM_CheckNearlyEQOp : VM_BinaryCheckOp<"check.nearly_eq", [Commutative]> {
  let hasCanonicalizer = 1;
}

def VM_CheckNEOp : VM_BinaryCheckOp<"check.ne", [Commutative]> {
  let hasCanonicalizer = 1;
}

def VM_CheckNZOp : VM_UnaryCheckOp<"check.nz"> {
  let hasCanonicalizer = 1;
}

def VM_ImportResolvedOp : VM_TrivialOp<"import.resolved", [
  DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = [{returns true if an optional import was resolved at runtime}];
  let description = [{
    Allows for checking whether a optional import was resolved at runtime. If
    this returns false then attempting to call the imported function will result
    in a failure at runtime.
  }];

  let arguments = (ins
    VM_FuncRefAttr:$import
  );
  let results = (outs
    VM_CondValue:$result
  );

  let assemblyFormat = [{
    $import attr-dict `:` type($result)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_ImportResolved>,
    VM_EncFuncAttr<"import">,
    VM_EncResult<"result">,
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Async/fiber ops
//===----------------------------------------------------------------------===//

// TODO(benvanik): WaitHandle and wait functions with YieldPoint traits:
//   signaling ops
//   await_all
//   await_any

def VM_YieldOp : VM_Op<"yield", [
    DeclareOpInterfaceMethods<BranchOpInterface>,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    HasParent<"IREE::VM::FuncOp">,
    Terminator,
    Util_YieldPoint,
  ]> {
  let summary = [{unconditional fiber yield operation}];
  let description = [{
    Yields the fiber for some (likely short) amount of time. This can be used to
    perform cooperative scheduling and ensure fair (enough) execution. Execution
    resumes at the specified target branch.

    ```
    ^bb0:
      vm.yield ^on_resume
    ^on_resume:
      ...
   ```
  }];

  let arguments = (ins
    Variadic<VM_AnyType>:$destOperands
  );

  let successors = (successor
    AnySuccessor:$dest
  );

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Yield>,
    VM_EncBranch<"dest", "getOperands", 0>,
  ];

  let builders = [
    OpBuilder<(ins "Block *":$dest, CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>,
  ];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];
}

//===----------------------------------------------------------------------===//
// Debugging
//===----------------------------------------------------------------------===//

def VM_TraceOp : VM_Op<"trace", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    VM_FullBarrier,
    VM_DebugOnly,
  ]> {
  let summary = [{trace value(s) operation}];
  let description = [{
    Traces one or more values at the time the operation is executed.
    These values will be encoded into the active trace depending on the active
    trace verbosity setting.
  }];

  let arguments = (ins
    StrAttr:$event_name,
    Variadic<VM_AnyType>:$operands
  );

  let assemblyFormat = [{
    $event_name `(` operands `)` attr-dict `:` type(operands)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Trace>,
    VM_EncStrAttr<"event_name">,
    VM_EncVariadicOperands<"operands">,
  ];

  let hasCanonicalizer = 1;
}

def VM_PrintOp : VM_Op<"print", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    VM_FullBarrier,
    VM_DebugOnly,
  ]> {
  let summary = [{message printing operation}];
  let description = [{
    Prints the given string message and zero or more values.
  }];

  let arguments = (ins
    StrAttr:$message,
    Variadic<VM_AnyType>:$operands
  );

  let assemblyFormat = [{
    $message `(` operands `)` attr-dict `:` type(operands)
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Print>,
    VM_EncStrAttr<"message">,
    VM_EncVariadicOperands<"operands">,
  ];

  let hasCanonicalizer = 1;
}

def VM_BreakOp : VM_Op<"break", [
    DeclareOpInterfaceMethods<BranchOpInterface>,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    Terminator,
    Util_YieldPoint,
    VM_FullBarrier,
    VM_DebugOnly,
  ]> {
  let summary = [{unconditional debug break operation}];
  let description = [{
    Breaks into the attached debugger or asks for attaching a debugger. After
    resuming (or if a debugger is not attached) execution will continue at the
    target block.
  }];

  let arguments = (ins
    Variadic<VM_AnyType>:$destOperands
  );

  let successors = (successor
    AnySuccessor:$dest
  );

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_Break>,
    VM_EncBranch<"dest", "getOperands", 0>,
  ];

  let builders = [
    OpBuilder<(ins "Block *":$dest, CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>,
  ];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizer = 1;
}

// TODO(benvanik): remove and turn into a compare + branch + vm.break.
def VM_CondBreakOp : VM_Op<"cond_break", [
    DeclareOpInterfaceMethods<BranchOpInterface>,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    Terminator,
    Util_YieldPoint,
    VM_FullBarrier,
    VM_DebugOnly,
  ]> {
  let summary = [{conditional debug break operation}];
  let description = [{
    Breaks into the attached debugger or asks for attaching a debugger if the
    provided condition is true. After resuming (or if a debugger is not
    attached) execution will continue at the target block.
  }];

  let arguments = (ins
    VM_CondValue:$condition,
    Variadic<VM_AnyType>:$destOperands
  );

  let successors = (successor
    AnySuccessor:$dest
  );

  let assemblyFormat = [{
    $condition `,` $dest (`(` $destOperands^ `:` type($destOperands) `)`)?
    attr-dict
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_CondBreak>,
    VM_EncBranch<"dest", "getOperands", 0>,
  ];

  let builders = [
    OpBuilder<(ins "Value":$condition, "Block *":$dest,
      CArg<"ValueRange", "{}">:$destOperands),
    [{
      $_state.addOperands({condition});
      $_state.addOperands(destOperands);
      $_state.addSuccessors(dest);
    }]>,
  ];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_VM_OPS

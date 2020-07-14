// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_DIALECT_VM_OPS
#define IREE_DIALECT_VM_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "iree/compiler/Dialect/VM/IR/VMBase.td"

class VM_PureOp<string mnemonic, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;

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
    StrAttr:$sym_name,
    // TODO(benvanik): add compatibility and versioning attributes.
    OptionalAttr<DictionaryAttr>:$ordinal_counts
  );

  let regions = (region SizedRegion<1>:$body);

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &, OperationState &state, StringRef name
    }]>,
  ];

  let extraClassDeclaration = [{
    Region &getBodyRegion() { return getOperation()->getRegion(0); }
    Block& getBlock() { return this->getOperation()->getRegion(0).front(); }
  }];

  let verifier = [{ return verifyModuleOp(*this); }];
}

def VM_ModuleTerminatorOp : VM_Op<"module_terminator", [
    HasParent<"IREE::VM::ModuleOp">,
    Terminator,
  ]> {
  let summary = [{terminator pseudo-op for the module op}];

  let assemblyFormat = "attr-dict";
}

// TODO(b/142336293): DCE functions when unused and not exports.
// TODO(benvanik): attributes for function analysis: pure, etc (for imports).
def VM_FuncOp : VM_Op<"func", [
    IsolatedFromAbove,
    HasParent<"IREE::VM::ModuleOp">,
    NativeOpTrait<"FunctionLike">,
    CallableOpInterface,
    Symbol,
  ]> {
  let summary = [{function defined with VM control flow ops}];
  let description = [{
    Represents a function containing VM ops and those of compatible dialects.
    All flow control is performed by VM ops.
  }];

  let arguments = (ins
    OptionalAttr<VM_Ordinal>:$ordinal,
    OptionalAttr<UnitAttr>:$noinline
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      FunctionType type, ArrayRef<NamedAttribute> attrs = {},
      ArrayRef<MutableDictionaryAttr> argAttrs = {}
    }]>,
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

    /// Returns the type of this function.
    FunctionType getType() {
      return getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<FunctionType>();
    }

    /// Hook for OpTrait::FunctionLike, returns the number of function
    /// arguments. Depends on the type attribute being correct as checked by
    /// verifyType.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Hook for OpTrait::FunctionLike, returns the number of function results.
    /// Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return &body(); }
    ArrayRef<Type> getCallableResults() {
      assert(!isExternal() && "invalid callable");
      return getType().getResults();
    }
  }];
}

def VM_ExportOp : VM_Op<"export", [
    HasParent<"IREE::VM::ModuleOp">,
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
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, FuncOp functionRef,
      StringRef exportName = "", ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, FlatSymbolRefAttr functionRef,
      StringRef exportName = "", ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];
}

def VM_ImportOp : VM_Op<"import", [
    Symbol,
    NativeOpTrait<"FunctionLike">,
    CallableOpInterface,
  ]> {
  let summary = [{imports a function from an external module}];
  let description = [{
    Specifies a function that should be imported from either the runtime or
    an external VM module.
  }];

  let arguments = (ins
    OptionalAttr<VM_Ordinal>:$ordinal
  );

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      FunctionType type, ArrayRef<NamedAttribute> attrs = {},
      ArrayRef<MutableDictionaryAttr> argAttrs = {}
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Returns the type of this function.
    FunctionType getType() {
      return getAttrOfType<TypeAttr>(getTypeAttrName())
          .getValue()
          .cast<FunctionType>();
    }

    /// Returns true if the import must be invoked via vm.call.variadic.
    bool isVariadic() {
      for (int i = 0; i < getNumFuncArguments(); ++i) {
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

    /// Hook for OpTrait::FunctionLike, returns the number of function
    /// arguments. Depends on the type attribute being correct as checked by
    /// verifyType.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Hook for OpTrait::FunctionLike, returns the number of function results.
    /// Depends on the type attribute being correct as checked by verifyType.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present. This can check for preconditions of the
    /// getNumArguments hook not failing.
    LogicalResult verifyType();

    Region *getCallableRegion() { return nullptr; }
    ArrayRef<Type> getCallableResults() {
      return getType().getResults();
    }
  }];
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// TODO(b/142336293): DCE globals when unused and not exports. Note that side
// effects may need to be observed.
class VM_GlobalOp<string mnemonic, Attr attr_type, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      IsolatedFromAbove,
      HasParent<"IREE::VM::ModuleOp">,
      Symbol,
    ])> {
  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttr:$type,
    UnitAttr:$is_mutable,
    OptionalAttr<FlatSymbolRefAttr>:$initializer,
    OptionalAttr<attr_type>:$initial_value,
    OptionalAttr<VM_Ordinal>:$ordinal
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type,
      Optional<StringRef> initializer, Optional<Attribute> initialValue,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, IREE::VM::FuncOp initializer,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type, Attribute initialValue,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      bool isMutable, Type type, ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];

  let extraClassDeclaration = [{
    void makeMutable() { setAttr("is_mutable", UnitAttr::get(getContext())); }
    void clearInitializer() { removeAttr("initializer"); }
    void clearInitialValue() { removeAttr("initial_value"); }
  }];

  let parser = [{ return parseGlobalOp(parser, &result); }];
  let printer = [{ return printGlobalOp(p, *this); }];
  let verifier = [{ return verifyGlobalOp(*this); }];
}

def VM_GlobalI32Op : VM_GlobalOp<"global.i32", VM_ConstIntValueAttr<I32>> {
  let summary = [{32-bit integer global declaration}];
  let description = [{
    Defines a global value that is treated as a scalar literal at runtime.
    Initialized to zero unless a custom initializer function is specified.
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalRefOp : VM_GlobalOp<"global.ref", UnitAttr> {
  let summary = [{ref_ptr<T> global declaration}];
  let description = [{
    Defines a global value that is a ref_ptr of a specific type. The global will
    retain the ref object for the lifetime of the context or until the value is
    replaced with a store or reset.
  }];

  let hasCanonicalizer = 1;
}

def VM_GlobalAddressOp : VM_PureOp<"global.address"> {
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
    AnyPtr:$result
  );

  let assemblyFormat = "$global attr-dict `:` type($result)";

  let verifier = [{ return verifyGlobalAddressOp(*this); }];
}

class VM_GlobalLoadOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let arguments = (ins
    VM_GlobalRefAttr:$global
  );
  let results = (outs
    type:$value
  );

  let assemblyFormat = "$global attr-dict `:` type($value)";

  let verifier = [{ return verifyGlobalLoadOp(*this); }];
}

class VM_GlobalStoreOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let arguments = (ins
    type:$value,
    VM_GlobalRefAttr:$global
  );

  let assemblyFormat = "$value `,` $global attr-dict `:` type($value)";

  let verifier = [{ return verifyGlobalStoreOp(*this); }];
}

class VM_GlobalLoadIndirectOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let arguments = (ins
    AnyTypeOf<[VM_Ptr, PtrOf<type>]>:$global
  );
  let results = (outs
    type:$value
  );

  let assemblyFormat = "$global attr-dict `:` type($global) `->` type($value)";
}

class VM_GlobalStoreIndirectOp<Type type, string mnemonic, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let arguments = (ins
    type:$value,
    AnyTypeOf<[VM_Ptr, PtrOf<type>]>:$global
  );

  let assemblyFormat = "$value `,` $global attr-dict `:` type($value) `->` type($global)";
}

def VM_GlobalLoadI32Op : VM_GlobalLoadOp<I32, "global.load.i32"> {
  let summary = [{global 32-bit integer load operation}];
  let description = [{
    Loads the value of a global containing a 32-bit integer.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalLoadI32>,
    VM_EncGlobalAttr<"global">,
    VM_EncResult<"value">,
  ];

  let hasCanonicalizer = 1;
}

def VM_GlobalStoreI32Op : VM_GlobalStoreOp<I32, "global.store.i32"> {
  let summary = [{global 32-bit integer store operation}];
  let description = [{
    Stores the 32-bit integer value to a global.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalStoreI32>,
    VM_EncGlobalAttr<"global">,
    VM_EncOperand<"value", 0>,
  ];
}

def VM_GlobalLoadIndirectI32Op :
    VM_GlobalLoadIndirectOp<I32, "global.load.indirect.i32"> {
  let summary = [{global 32-bit integer load operation}];
  let description = [{
    Loads the value of a global containing a 32-bit integer.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalLoadIndirectI32>,
    VM_EncOperand<"global", 0>,
    VM_EncResult<"value">,
  ];

  let hasCanonicalizer = 1;
}

def VM_GlobalStoreIndirectI32Op :
    VM_GlobalStoreIndirectOp<I32, "global.store.indirect.i32"> {
  let summary = [{global 32-bit integer store operation}];
  let description = [{
    Stores the 32-bit integer value to a global.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalStoreIndirectI32>,
    VM_EncOperand<"global", 0>,
    VM_EncOperand<"value", 1>,
  ];

  let hasCanonicalizer = 1;
}

def VM_GlobalLoadRefOp : VM_GlobalLoadOp<VM_AnyRef, "global.load.ref"> {
  let summary = [{global ref_ptr<T> load operation}];
  let description = [{
    Loads the value of a global containing a ref_ptr of the given type.
  }];

  let encoding = [
    VM_EncOpcode<VM_OPC_GlobalLoadRef>,
    VM_EncGlobalAttr<"global">,
    VM_EncTypeOf<"value">,
    VM_EncResult<"value">,
  ];

  let hasCanonicalizer = 1;
}

def VM_GlobalStoreRefOp : VM_GlobalStoreOp<VM_AnyRef, "global.store.ref"> {
  let summary = [{global ref_ptr<T> stores operation}];
  let description = [{
    Stores a ref_ptr<T> to a global, retaining it until the global is reset.
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
  let summary = [{global ref_ptr<T> load operation}];
  let description = [{
    Loads the value of a global containing a ref_ptr of the given type.
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
  let summary = [{global ref_ptr<T> stores operation}];
  let description = [{
    Stores a ref_ptr<T> to a global, retaining it until the global is reset.
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

class VM_ConstOp<string mnemonic, string ctype, list<OpTrait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      ConstantLike,
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Attribute value
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, }] # ctype # [{ value
    }]>,
  ];

  let extraClassDeclaration = [{
    /// Returns true if a constant operation can be built with the given value
    /// and result type.
    static bool isBuildableWith(Attribute value, Type type);

    /// Returns an attribute in the appropriate type for the const op.
    static Attribute convertConstValue(Attribute value);
  }];
}

class VM_ConstIntegerOp<I type, string mnemonic, VM_OPC opcode, string ctype,
                        list<OpTrait> traits = []> :
    VM_ConstOp<mnemonic, ctype, traits> {
  let description = [{
    Defines a constant value that is treated as a scalar literal at runtime.
  }];

  let arguments = (ins
    VM_ConstIntValueAttr<type>:$value
  );
  let results = (outs
    type:$result
  );

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncIntAttr<"value", type.bitwidth>,
    VM_EncResult<"result">,
  ];
}

def VM_ConstI32Op :
    VM_ConstIntegerOp<I32, "const.i32", VM_OPC_ConstI32, "int32_t"> {
  let summary = [{32-bit integer constant operation}];
  let hasFolder = 1;
}

def VM_ConstI32ZeroOp : VM_PureOp<"const.i32.zero", [
    ConstantLike,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
  ]> {
  let summary = [{32-bit integer constant zero operation}];
  let description = [{
    Defines a constant zero 32-bit integer.
  }];

  let results = (outs
    I32:$result
  );

  let assemblyFormat = "`:` type($result) attr-dict";

  let encoding = [
    VM_EncOpcode<VM_OPC_ConstI32Zero>,
    VM_EncResult<"result">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result
    }]>,
  ];

  let hasFolder = 1;
}

def VM_ConstRefZeroOp : VM_PureOp<"const.ref.zero", [
    ConstantLike,
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
  ]> {
  let summary = [{null ref_ptr constant operation}];
  let description = [{
    Defines a constant null ref_ptr that can be used in comparisons and
    initialization.
  }];

  let results = (outs
    VM_AnyRef:$result
  );

  let assemblyFormat = "`:` type($result) attr-dict";

  let encoding = [
    VM_EncOpcode<VM_OPC_ConstRefZero>,
    VM_EncResult<"result">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Type objectType
    }]>,
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
    ref_ptr. This can be used to store arbitrary data within modules such as
    large constant buffers and other file contents.

    Note that the data is reference counted as a way to track its usage once the
    value leaves the module. For example, returning rodata from an exported
    function must keep the data (possibly backed by mmap) valid for its entire
    lifetime.
  }];

  let arguments = (ins
    StrAttr:$sym_name,
    ElementsAttr:$value,
    OptionalAttr<VM_Ordinal>:$ordinal
  );

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef name,
      ElementsAttr value, ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];
}

def VM_ConstRefRodataOp : VM_PureOp<"const.ref.rodata", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
  ]> {
  let summary = [{constant rodata access operation}];
  let description = [{
    Returns a reference to a read-only buffer.
  }];

  let arguments = (ins
    ByteBufferRefAttr:$rodata
  );

  let results = (outs
    VM_RefOf<ByteBufferType>:$value
  );

  let assemblyFormat = "$rodata attr-dict `:` type($value)";

  let encoding = [
    VM_EncOpcode<VM_OPC_ConstRefRodata>,
    VM_EncRodataAttr<"rodata">,
    VM_EncResult<"value">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef rodataName,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, RodataOp rodataOp,
      ArrayRef<NamedAttribute> attrs = {}
    }]>,
  ];

  let verifier = [{ return verify$cppClass(*this); }];
}

//===----------------------------------------------------------------------===//
// Lists
//===----------------------------------------------------------------------===//

// TODO(benvanik): vm.list.push.i32 / vm.list.pop.i32 (variadic)
// TODO(benvanik): vm.list.copy(src_list, src_index, dst_list, dst_index, length)
// TODO(benvanik): vm.list.slice(list, index, length) -> list
// TODO(benvanik): export into a rwdata buffer

def VM_ListAllocOp :
    VM_PureOp<"list.alloc", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{allocates a new empty list}];
  let description = [{
    Allocates a new typed list with a minimum initial_capacity.
  }];

  let arguments = (ins
    I32:$initial_capacity
  );
  let results = (outs
    VM_AnyList:$result
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($initial_capacity) `)` `->` type($result)";

  let encoding = [
    VM_EncOpcode<VM_OPC_ListAlloc>,
    VM_EncType<"result().getType().cast<IREE::VM::RefType>().getObjectType().cast<IREE::VM::ListType>().getElementType()">,
    VM_EncOperand<"initial_capacity", 0>,
    VM_EncResult<"result">,
  ];
}

def VM_ListReserveOp :
    VM_Op<"list.reserve", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{reserves capacity for list growth}];
  let description = [{
    Reserves storage for at least minimum_capacity elements. If the list already
    has at least the specified capacity the operation is ignored.
  }];

  let arguments = (ins
    VM_AnyList:$list,
    I32:$minimum_capacity
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `,` type($minimum_capacity) `)`";

  let encoding = [
    VM_EncOpcode<VM_OPC_ListReserve>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"minimum_capacity", 1>,
  ];
}

def VM_ListSizeOp :
    VM_Op<"list.size", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{the size of the list in elements}];
  let description = [{
    Returns the current size of the list in elements.
  }];

  let arguments = (ins
    VM_AnyList:$list
  );
  let results = (outs
    I32:$result
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `)` `->` type($result)";

  let encoding = [
    VM_EncOpcode<VM_OPC_ListSize>,
    VM_EncOperand<"list", 0>,
    VM_EncResult<"result">,
  ];
}

def VM_ListResizeOp :
    VM_Op<"list.resize", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{resizes the list to a new count in elements}];
  let description = [{
    Resizes the list to contain new_size elements. This will either truncate
    the list if the existing size is greater than new_size or extend the list
    with the default list value of 0 if storing primitives and null if refs.
  }];

  let arguments = (ins
    VM_AnyList:$list,
    I32:$new_size
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `,` type($new_size) `)`";

  let encoding = [
    VM_EncOpcode<VM_OPC_ListResize>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"new_size", 1>,
  ];
}

class VM_ListGetPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                            list<OpTrait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let summary = [{primitive type element accessor}];
  let description = [{
    Returns the value of the element at the given index.
  }];

  let arguments = (ins
    VM_ListOf<VM_PrimitiveType>:$list,
    I32:$index
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `,` type($index) `)` `->` type($result)";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncResult<"result">,
  ];
}

class VM_ListSetPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                            list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ])> {
  let summary = [{primitive type element mutator}];
  let description = [{
    Sets the element at the given index to the new value.
  }];

  let arguments = (ins
    VM_ListOf<VM_PrimitiveType>:$list,
    I32:$index,
    type:$value
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `,` type($index) `,` type($value) `)`";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncOperand<"value", 2>,
  ];
}

def VM_ListGetI32Op :
    VM_ListGetPrimitiveOp<I32, "list.get.i32", VM_OPC_ListGetI32> {}

def VM_ListSetI32Op :
    VM_ListSetPrimitiveOp<I32, "list.set.i32", VM_OPC_ListSetI32> {}

def VM_ListGetRefOp :
    VM_PureOp<"list.get.ref", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{ref type element accessor}];
  let description = [{
    Returns the ref value of the element at the given index. Note that the value
    may be null if the element is null or the type does not match.
  }];

  let arguments = (ins
    VM_AnyList:$list,
    I32:$index
  );
  let results = (outs
    VM_AnyRef:$result
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `,` type($index) `)` `->` type($result)";

  let encoding = [
    VM_EncOpcode<VM_OPC_ListGetRef>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncTypeOf<"result">,
    VM_EncResult<"result">,
  ];

  let verifier = [{ return verify$cppClass(*this); }];
}

def VM_ListSetRefOp :
    VM_Op<"list.set.ref", [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    ]> {
  let summary = [{ref type element mutator}];
  let description = [{
    Sets the element at the given index to the new ref value (possibly null).
  }];

  let arguments = (ins
    VM_AnyList:$list,
    I32:$index,
    VM_AnyRef:$value
  );

  let assemblyFormat = "operands attr-dict `:` `(` type($list) `,` type($index) `,` type($value) `)`";

  let encoding = [
    VM_EncOpcode<VM_OPC_ListSetRef>,
    VM_EncOperand<"list", 0>,
    VM_EncOperand<"index", 1>,
    VM_EncOperand<"value", 2>,
  ];

  let verifier = [{ return verify$cppClass(*this); }];
}

//===----------------------------------------------------------------------===//
// Conditional assignment
//===----------------------------------------------------------------------===//

class VM_SelectPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                           list<OpTrait> traits = []> :
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

  let assemblyFormat = "operands attr-dict `:` type($result)";

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

def VM_SelectRefOp : VM_PureOp<"select.ref", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    AllTypesMatch<["true_value", "false_value", "result"]>,
  ]> {
  let summary = [{ref_ptr select operation}];
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

  let assemblyFormat = "operands attr-dict `:` type($result)";

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

class VM_SwitchPrimitiveOp<Type type, string mnemonic, VM_OPC opcode,
                           list<OpTrait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["default_value", "result"]>,
    ])> {
  let description = [{
    Returns the value with the given `index` in `values` or `default_value` if
    the index is out of bounds.

    ```mlir
    // Switch %arg0 to cases of %c100/%c200/%c300 if arg0==0, ==1, ==2.
    // If %arg0 is out of range (<0 or >2) then default to %c5.
    %0 = vm.switch.i32 %index[%c100, %c200, %c300] else %c5 : i32
    ```
  }];

  let arguments = (ins
    I32:$index,
    I32:$default_value,
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
    VM_EncIntAttr<"default_value", 32>,
    VM_EncVariadicOperands<"values">,
    VM_EncResult<"result">,
  ];
}

def VM_SwitchI32Op : VM_SwitchPrimitiveOp<I32, "switch.i32", VM_OPC_SwitchI32> {
  let summary = [{integer switch operation}];
  let hasFolder = 1;
}

def VM_SwitchRefOp : VM_PureOp<"switch.ref", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    AllTypesMatch<["default_value", "result"]>,
  ]> {
  let summary = [{ref_ptr switch operation}];
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
    I32:$index,
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
// Native integer arithmetic
//===----------------------------------------------------------------------===//

class VM_UnaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                           list<OpTrait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["operand", "result"]>,
    ])> {
  let arguments = (ins
    type:$operand
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = "$operand attr-dict `:` type($result)";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncResult<"result">,
  ];
}

class VM_BinaryArithmeticOp<Type type, string mnemonic, VM_OPC opcode,
                            list<OpTrait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
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

  let assemblyFormat = "operands attr-dict `:` type($result)";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"lhs", 0>,
    VM_EncOperand<"rhs", 1>,
    VM_EncResult<"result">,
  ];
}

def VM_AddI32Op :
    VM_BinaryArithmeticOp<I32, "add.i32", VM_OPC_AddI32, [Commutative]> {
  let summary = [{integer add operation}];
  let hasFolder = 1;
}

def VM_SubI32Op :
    VM_BinaryArithmeticOp<I32, "sub.i32", VM_OPC_SubI32> {
  let summary = [{integer subtract operation}];
  let hasFolder = 1;
}

def VM_MulI32Op :
    VM_BinaryArithmeticOp<I32, "mul.i32", VM_OPC_MulI32, [Commutative]> {
  let summary = [{integer multiplication operation}];
  let hasFolder = 1;
}

def VM_DivI32SOp :
    VM_BinaryArithmeticOp<I32, "div.i32.s", VM_OPC_DivI32S> {
  let summary = [{signed integer division operation}];
  let hasFolder = 1;
}

def VM_DivI32UOp :
    VM_BinaryArithmeticOp<I32, "div.i32.u", VM_OPC_DivI32U> {
  let summary = [{unsigned integer division operation}];
  let hasFolder = 1;
}

def VM_RemI32SOp :
    VM_BinaryArithmeticOp<I32, "rem.i32.s", VM_OPC_RemI32S> {
  let summary = [{signed integer division remainder operation}];
  let hasFolder = 1;
}

def VM_RemI32UOp :
    VM_BinaryArithmeticOp<I32, "rem.i32.u", VM_OPC_RemI32U> {
  let summary = [{unsigned integer division remainder operation}];
  let hasFolder = 1;
}

def VM_NotI32Op :
    VM_UnaryArithmeticOp<I32, "not.i32", VM_OPC_NotI32> {
  let summary = [{integer binary not operation}];
  let hasFolder = 1;
}

def VM_AndI32Op :
    VM_BinaryArithmeticOp<I32, "and.i32", VM_OPC_AndI32, [Commutative]> {
  let summary = [{integer binary and operation}];
  let hasFolder = 1;
}

def VM_OrI32Op :
    VM_BinaryArithmeticOp<I32, "or.i32", VM_OPC_OrI32, [Commutative]> {
  let summary = [{integer binary or operation}];
  let hasFolder = 1;
}

def VM_XorI32Op :
    VM_BinaryArithmeticOp<I32, "xor.i32", VM_OPC_XorI32, [Commutative]> {
  let summary = [{integer binary exclusive-or operation}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Native bitwise shifts and rotates
//===----------------------------------------------------------------------===//

class VM_ShiftArithmeticOp<I type, string mnemonic, VM_OPC opcode,
                           list<OpTrait> traits = []> :
    VM_PureOp<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      AllTypesMatch<["operand", "result"]>,
    ])> {
  let description = [{
    Shifts the operand in a direction by the number of bits specified.
  }];

  let arguments = (ins
    type:$operand,
    Confined<I8Attr, [IntegerAttrInRange<0, type.bitwidth>]>:$amount
  );
  let results = (outs
    type:$result
  );

  let assemblyFormat = "$operand `,` $amount attr-dict `:` type($operand)";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncIntAttr<"amount", type.bitwidth>,
    VM_EncResult<"result">,
  ];
}

def VM_ShlI32Op : VM_ShiftArithmeticOp<I32, "shl.i32", VM_OPC_ShlI32> {
  let summary = [{integer shift left operation}];
  let hasFolder = 1;
}

def VM_ShrI32SOp : VM_ShiftArithmeticOp<I32, "shr.i32.s", VM_OPC_ShrI32S> {
  let summary = [{signed integer (arithmetic) shift right operation}];
  let hasFolder = 1;
}

def VM_ShrI32UOp : VM_ShiftArithmeticOp<I32, "shr.i32.u", VM_OPC_ShrI32U> {
  let summary = [{unsigned integer (logical) shift right operation}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Casting and type conversion/emulation
//===----------------------------------------------------------------------===//

def VM_TruncI8Op : VM_UnaryArithmeticOp<I32, "trunc.i8", VM_OPC_TruncI8> {
  let summary = [{integer truncate to 8 bits}];
  let hasFolder = 1;
}

def VM_TruncI16Op : VM_UnaryArithmeticOp<I32, "trunc.i16", VM_OPC_TruncI16> {
  let summary = [{integer truncate to 16 bits}];
  let hasFolder = 1;
}

def VM_ExtI8I32SOp : VM_UnaryArithmeticOp<I32, "ext.i8.i32.s", VM_OPC_ExtI8I32S> {
  let summary = [{integer sign extend 8 bits to 32 bits}];
  let hasFolder = 1;
}

def VM_ExtI16I32SOp : VM_UnaryArithmeticOp<I32, "ext.i16.i32.s", VM_OPC_ExtI16I32S> {
  let summary = [{integer sign extend 16 bits to 32 bits}];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Native reduction (horizontal) arithmetic
//===----------------------------------------------------------------------===//

// TODO(benvanik): used for stride/element count/etc.

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

class VM_UnaryComparisonOp<Type type, string mnemonic, VM_OPC opcode,
                           list<OpTrait> traits = []> :
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
    I32:$result
  );

  let assemblyFormat = "$operand attr-dict `:` type($operand)";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"operand", 0>,
    VM_EncResult<"result">,
  ];
}

class VM_BinaryComparisonOp<Type type, string mnemonic, VM_OPC opcode,
                            list<OpTrait> traits = []> :
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
    I32:$result
  );

  let assemblyFormat = "operands attr-dict `:` type($lhs)";

  let encoding = [
    VM_EncOpcode<opcode>,
    VM_EncOperand<"lhs", 0>,
    VM_EncOperand<"rhs", 1>,
    VM_EncResult<"result">,
  ];
}

def VM_CmpEQI32Op :
    VM_BinaryComparisonOp<I32, "cmp.eq.i32", VM_OPC_CmpEQI32, [Commutative]> {
  let summary = [{integer equality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNEI32Op :
    VM_BinaryComparisonOp<I32, "cmp.ne.i32", VM_OPC_CmpNEI32, [Commutative]> {
  let summary = [{integer inequality comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTI32SOp :
    VM_BinaryComparisonOp<I32, "cmp.lt.i32.s", VM_OPC_CmpLTI32S> {
  let summary = [{signed integer less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTI32UOp :
    VM_BinaryComparisonOp<I32, "cmp.lt.i32.u", VM_OPC_CmpLTI32U> {
  let summary = [{unsigned integer less-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEI32SOp :
    VM_BinaryComparisonOp<I32, "cmp.lte.i32.s", VM_OPC_CmpLTEI32S> {
  let summary = [{signed integer less-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpLTEI32UOp :
    VM_BinaryComparisonOp<I32, "cmp.lte.i32.u", VM_OPC_CmpLTEI32U> {
  let summary = [{unsigned integer less-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

// TODO(benvanik): drop these and rely on lt/lte only?
def VM_CmpGTI32SOp :
    VM_BinaryComparisonOp<I32, "cmp.gt.i32.s", VM_OPC_CmpGTI32S> {
  let summary = [{signed integer greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTI32UOp :
    VM_BinaryComparisonOp<I32, "cmp.gt.i32.u", VM_OPC_CmpGTI32U> {
  let summary = [{unsigned integer greater-than comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEI32SOp :
    VM_BinaryComparisonOp<I32, "cmp.gte.i32.s", VM_OPC_CmpGTEI32S> {
  let summary = [{signed integer greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpGTEI32UOp :
    VM_BinaryComparisonOp<I32, "cmp.gte.i32.u", VM_OPC_CmpGTEI32U> {
  let summary = [{unsigned integer greater-than-or-equal comparison operation}];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def VM_CmpNZI32Op :
    VM_UnaryComparisonOp<I32, "cmp.nz.i32", VM_OPC_CmpNZI32> {
  let summary = [{integer non-zero comparison operation}];
  let description = [{
    Compares the given integer operand for a non-zero value.
  }];
  let hasFolder = 1;
}

def VM_CmpEQRefOp :
    VM_BinaryComparisonOp<VM_AnyRef, "cmp.eq.ref", VM_OPC_CmpEQRef,
                          [Commutative]> {
  let summary = [{ref_ptr equality comparison operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// TODO(benvanik): remove and just rely on eq & flipped cond_br?
def VM_CmpNERefOp :
    VM_BinaryComparisonOp<VM_AnyRef, "cmp.ne.ref", VM_OPC_CmpNERef,
                          [Commutative]> {
  let summary = [{ref_ptr inequality comparison operation}];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def VM_CmpNZRefOp :
    VM_UnaryComparisonOp<VM_AnyRef, "cmp.nz.ref", VM_OPC_CmpNZRef> {
  let summary = [{ref_ptr non-zero comparison operation}];
  let description = [{
    Compares the given ref_ptr operand for a non-zero/null value.
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
    OpBuilder<[{
      OpBuilder &, OperationState &result, Block *dest,
      ValueRange destOperands = {}
    }], [{
      result.addSuccessors(dest);
      result.addOperands(destOperands);
  }]>];

  let extraClassDeclaration = [{
    Block *getDest();
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
    VM_EncBranch<"getTrueDest", "getTrueOperands", 0>,
    VM_EncBranch<"getFalseDest", "getFalseOperands", 1>,
  ];

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, Value condition,
      Block *trueDest, ValueRange trueOperands,
      Block *falseDest, ValueRange falseOperands
    }], [{
      build(builder, result, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    }]>, OpBuilder<[{
      OpBuilder &builder, OperationState &result, Value condition,
      Block *trueDest, Block *falseDest, ValueRange falseOperands = {}
    }], [{
      build(builder, result, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    }]>
  ];

  let extraClassDeclaration = [{
    /// These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    /// The condition operand is the first operand in the list.
    Value getCondition() { return getOperand(0); }

    /// Return the destination if the condition is true.
    Block *getTrueDest() {
      return getOperation()->getSuccessor(trueIndex);
    }

    /// Return the destination if the condition is false.
    Block *getFalseDest() {
      return getOperation()->getSuccessor(falseIndex);
    }

    /// Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return trueDestOperands(); }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      trueDestOperandsMutable().erase(index);
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

    operand_range getFalseOperands() { return falseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      falseDestOperandsMutable().erase(index);
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

class VM_CallBaseOp<string mnemonic, list<OpTrait> traits = []> :
    VM_Op<mnemonic, !listconcat(traits, [
      DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
      CallOpInterface,
    ])> {
  let extraClassDeclaration = [{
    // TODO(b/132296600): make tablegen follow the style guide.
    StringRef getCallee() { return callee(); }

    // TODO(b/133879130): make tablegen support variadic operand accessors.
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return getAttrOfType<FlatSymbolRefAttr>("callee");
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
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, IREE::VM::FuncOp callee,
      ValueRange operands = {}
    }], [{
      result.addOperands(operands);
      result.addAttribute("callee", builder.getSymbolRefAttr(callee));
      result.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, FlatSymbolRefAttr callee,
      ArrayRef<Type> resultTypes, ValueRange operands = {}
    }], [{
      result.addOperands(operands);
      result.addAttribute("callee", callee);
      result.addTypes(resultTypes);
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, StringRef callee,
      ArrayRef<Type> resultTypes, ValueRange operands = {}
    }], [{
      build(builder, result, builder.getSymbolRefAttr(callee),
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
    VM_EncIntArrayAttr<"segment_sizes", 16>,
    VM_EncVariadicOperands<"operands">,
    VM_EncVariadicResults<"results">,
  ];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result, FlatSymbolRefAttr callee,
      ArrayRef<Type> resultTypes, ArrayRef<int16_t> segmentSizes = {},
      ArrayRef<Type> segmentTypes = {}, ValueRange operands = {}
    }], [{
      result.addAttribute("callee", callee);
      result.addAttribute("segment_sizes",
          DenseIntElementsAttr::get(
              VectorType::get({static_cast<int64_t>(segmentSizes.size())},
                              builder.getIntegerType(16)),
              segmentSizes));
      result.addAttribute("segment_types", builder.getArrayAttr(
          llvm::to_vector<4>(llvm::map_range(segmentTypes, [&](Type type) {
              return TypeAttr::get(type).cast<Attribute>();
          }))));
      result.addOperands(operands);
      result.addTypes(resultTypes);
    }]>,
  ];

  let hasCanonicalizer = 1;
}

def VM_ReturnOp : VM_Op<"return", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    HasParent<"IREE::VM::FuncOp">,
    Terminator,
  ]> {
  let summary = "return operation";
  let description = [{
    Represents a return operation within a function.

    ```
    vm.func @foo(%0, %1) : (i32, f8) {
      vm.return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins
    Variadic<VM_AnyType>:$operands
  );

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";

  let encoding = [
    VM_EncOpcode<VM_OPC_Return>,
    VM_EncVariadicOperands<"operands">,
  ];

  let builders = [
    OpBuilder<[{
      OpBuilder &b, OperationState &result
    }], [{
      build(b, result, llvm::None);
    }]>,
  ];
}

def VM_FailOp : VM_Op<"fail", [
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    HasParent<"IREE::VM::FuncOp">,
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
    %statusCode = vm.const.i32 9 : i32
    vm.fail %statusCode, "oh no!"
    ```
  }];

  let arguments = (ins
    IREE_Status:$status,
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
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value status, StringRef message = ""
    }], [{
      build(
          builder, result, status,
          message.empty() ? StringAttr{} : builder.getStringAttr(message));
    }]>,
  ];

  let verifier = [{ return verifyFailOp(*this); }];
}

def VM_CondFailOp : VM_Op<"cond_fail", [
    HasParent<"IREE::VM::FuncOp">,
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
    %statusCode = vm.const.i32 9 : i32
    vm.cond_fail %nz, %statusCode, "expected non-zero"
    ```
  }];

  let arguments = (ins
    VM_CondValue:$condition,
    IREE_Status:$status,
    OptionalAttr<StrAttr>:$message
  );

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value condition, Value status, StringRef message = ""
    }], [{
      build(
          builder, result, condition, status,
          message.empty() ? StringAttr{} : builder.getStringAttr(message));
    }]>,
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value status, StringRef message = ""
    }], [{
      build(builder, result, status, status, message);
    }]>,
  ];

  let verifier = [{ return verifyFailOp(*this); }];

  let hasCanonicalizer = 1;
}

class VM_CheckOp<string mnemonic, list<OpTrait> traits = []> :
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

class VM_UnaryCheckOp<string mnemonic, list<OpTrait> traits = []> :
    VM_CheckOp<mnemonic, traits> {
  let arguments = (ins
    VM_AnyType:$value,
    OptionalAttr<StrAttr>:$message
  );


  let assemblyFormat = [{
    $value (`,` $message^)? attr-dict `:` type($value)
  }];

  let builders = [
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value value, StringRef message = ""
    }], [{
      build(
          builder, result, value,
          message.empty() ? StringAttr{} : builder.getStringAttr(message));
    }]>,
  ];
}

class VM_BinaryCheckOp<string mnemonic, list<OpTrait> traits = []> :
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
    OpBuilder<[{
      OpBuilder &builder, OperationState &result,
      Value lhs, Value rhs, StringRef message = ""
    }], [{
      build(
          builder, result, lhs, rhs,
          message.empty() ? StringAttr{} : builder.getStringAttr(message));
    }]>,
  ];
}

def VM_CheckEQOp : VM_BinaryCheckOp<"check.eq", [Commutative]> {
  let hasCanonicalizer = 1;
}

def VM_CheckNEOp : VM_BinaryCheckOp<"check.ne", [Commutative]> {
  let hasCanonicalizer = 1;
}

def VM_CheckNZOp : VM_UnaryCheckOp<"check.nz"> {
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
    DeclareOpInterfaceMethods<VM_SerializableOpInterface>,
    HasParent<"IREE::VM::FuncOp">,
    YieldPoint,
  ]> {
  let summary = [{unconditional fiber yield operation}];
  let description = [{
    Yields the fiber for some (likely short) amount of time. This can be used to
    perform cooperative scheduling and ensure fair (enough) execution.
  }];

  let assemblyFormat = "attr-dict";

  let encoding = [
    VM_EncOpcode<VM_OPC_Yield>,
  ];
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
    YieldPoint,
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
    OpBuilder<[{
      OpBuilder &, OperationState &result, Block *dest,
      ValueRange destOperands = {}
    }], [{
      result.addSuccessors(dest);
      result.addOperands(destOperands);
  }]>];

  let extraClassDeclaration = [{
    Block *getDest();
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
    YieldPoint,
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
    OpBuilder<[{
      OpBuilder &, OperationState &result, Value condition, Block *dest,
      ValueRange destOperands = {}
    }], [{
      result.addOperands({condition});
      result.addOperands(destOperands);
      result.addSuccessors(dest);
    }]>,
  ];

  let extraClassDeclaration = [{
    Block *getDest();
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizer = 1;
}

#endif  // IREE_DIALECT_VM_OPS

// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_DIALECT_TD
#define IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_DIALECT_TD

include "iree-dialects/Dialect/PyDM/IR/PyDMBase.td"
include "iree-dialects/Dialect/PyDM/IR/PyDMInterfaces.td"

//===----------------------------------------------------------------------===//
// Variable Types
//===----------------------------------------------------------------------===//

def IREEPyDM_FreeVarRef : IREEPyDM_TypeDef<"FreeVarRef"> {
  let mnemonic = "free_var_ref";
  let summary = "A direct access 'free' variable slot within a func";
  let description = [{
    A 'free variable' is a non-shared, direct access variable in a function.
    Other languages would call these 'locals'. In Python, they are distinguished
    from 'cell variables'.
  }];
}

def IREEPyDM_AnyVarRef : AnyTypeOf<[
  IREEPyDM_FreeVarRef,
], "Python variable reference">;

//===----------------------------------------------------------------------===//
// Unboxed Primitive Types
//===----------------------------------------------------------------------===//

// Declare a new primitive type.
// When adding a new one, update the PrimitiveType::classof method in
// IREEPyDMDialect.h.
class IREEPyDM_PrimitiveTypeDef<
  string name, list<string> overridenMethods = []> :
    TypeDef<IREEPyDM_Dialect, name, /*traits=*/[
      DeclareTypeInterfaceMethods<PythonTypeInterface, overridenMethods>
    ],
    /*baseCppClass=*/"::mlir::iree_compiler::IREE::PYDM::PrimitiveType"> {
}

def IREEPyDM_AnyPrimitiveType : Type<
    CPred<"$_self.isa<::mlir::iree_compiler::IREE::PYDM::PrimitiveType>()">,
    "unboxed primitive type",
    "::mlir::iree_compiler::IREE::PYDM::PrimitiveType">;

def IREEPyDM_BoolType : IREEPyDM_PrimitiveTypeDef<"Bool", [
    "getNumericPromotionOrder", "getNumericCategory", "getNumericSubTypeCode"]> {
  let mnemonic = "bool";

  let summary = "Type of bool values";

  let description = [{
    Represents boolean types in the data model, with values 'True' and 'False'.
    Note that the data model considers the bool type to be a subtype of
    integer, which is important during numeric promotion.
  }];
}

def IREEPyDM_BytesType : IREEPyDM_PrimitiveTypeDef<"Bytes"> {
  let mnemonic = "bytes";

  let summary = "Type representing byte strings";

  let description = [{
    Represent Pythong 'bytes'.
  }];
}

def IREEPyDM_ExceptionResultType : IREEPyDM_PrimitiveTypeDef<"ExceptionResult"> {
  let mnemonic = "exception_result";

  let summary = "Either successful or exceptional result";

  let description = [{
    The exception result connotes a logical success/failure state and can
    also carry additional user-level exception data. It is used as a return
    value from functions and many failable operations. Boxing a successful
    exception result produces a None object. Boxing a failed result produces
    an exception object.
  }];
}

def IREEPyDM_IntegerType : IREEPyDM_PrimitiveTypeDef<"Integer", [
    "getNumericPromotionOrder", "getNumericCategory", "getNumericSubTypeCode"]> {
  let mnemonic = "integer";

  let summary = "Type of integer values";

  let description = [{
    Represents the `numbers.Integral` type in the data model. Without further
    qualification, the type is considered weak and open to further inference.
    It can be further qualified to an explicit type:
      Signed of a given bitwidth
      Unsigned of a given bitwidth
      Arbitrary precision

    Prints as:
      integer : Weak integer
      integer<32> : Signed integer of specific bit width
      integer<unsigned 32> : Unsigned integer of specific bit width
      integer<*> : Arbitrary precision integer
  }];

  let parameters = (ins
    // Encodes:
    //   None: Weak integer
    //   0   : Arbitrary precision integer
    //   >0  : Signed
    //   <0  : Unsigned
    "Optional<int>":$bitWidth
  );

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    // Builds a weak integer.
    TypeBuilder<(ins), [{
      return Base::get($_ctxt, None);
    }]>,
    // Builds:
    //   (None): Arbitrary precision integer
    //   (32 [, true]): Signed integer of explicit size
    //   (32, false): Unsigned integer of explicit size
    TypeBuilder<(ins CArg<"Optional<unsigned>">:$bitWidth,
                     CArg<"bool", "true">:$isSigned), [{
      if (bitWidth) {
        // Explicit size
        int w = *bitWidth;
        if (!isSigned) w = -w;
        return Base::get($_ctxt, w);
      } else {
        // AP
        return Base::get($_ctxt, 0);
      }
    }]>,
  ];

  let genAccessors = 0;

  let extraClassDeclaration = [{
    bool isWeak() const;
    bool isExplicit() const { return !isWeak(); }
    unsigned getBitWidth() const;
    bool isSigned() const;
  }];

  let hasCustomAssemblyFormat = 1;
}

def IREEPyDM_ListType : IREEPyDM_PrimitiveTypeDef<"List", ["isRefinable"]> {
  let mnemonic = "list";

  let summary = "Mutable sequence of elements";

  let description = [{
    Corresponds to the "Lists" type in the data model.
  }];

  let parameters = (ins
    "CollectionStorageClass":$storageClass,
    "Type":$uniformElementType
  );

  let builders = [
    TypeBuilder<(ins), [{
      return Base::get($_ctxt, CollectionStorageClass::Boxed, nullptr);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Gets the type used to store elements in the backing list.
    /// For empty or boxed lists, this will be a generic ObjectType.
    /// For unboxed lists, it will be the uniform element type.
    Type getElementStorageType() const;
  }];
}

def IREEPyDM_NoneType : IREEPyDM_PrimitiveTypeDef<"None"> {
  let mnemonic = "none";

  let summary = "Type of the single 'None' value";

  let description = [{
    Represents the 'None' type in the standard type hierarchy.
  }];
}

def IREEPyDM_RealType : IREEPyDM_PrimitiveTypeDef<"Real",
    ["getNumericPromotionOrder", "getNumericCategory", "getNumericSubTypeCode"]> {
  let mnemonic = "real";

  let summary = "Type of floating point values";

  let description = [{
    Represents the `numbers.Real` type in the data model. Without qualification,
    the type is considered "weak" and left open to further inference and/or
    lowering defaults (which may differ from Python norms of treating this as
    an f64 if so configured).

    Prints as:
      `real` : Weak real type
      `real<f32>` : Explicit real type
  }];

  let parameters = (ins
    // Encodes:
    //   nullptr: Weak real
    //   FloatType: Explicit real of given floating point type
    "FloatType":$floatType
  );

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    // Builds a weak RealType
    TypeBuilder<(ins), [{
      return Base::get($_ctxt, nullptr);
    }]>,
    // Builds an explicit RealType
    TypeBuilder<(ins CArg<"FloatType">:$floatType), [{
      return Base::get($_ctxt, floatType);
    }]>,
  ];

  let extraClassDeclaration = [{
    bool isWeak() const;
    bool isExplicit() const { return !isWeak(); }
  }];
  let hasCustomAssemblyFormat = 1;
}

def IREEPyDM_StrType : IREEPyDM_PrimitiveTypeDef<"Str"> {
  let mnemonic = "str";

  let summary = "Type representing unicode strings";

  let description = [{
    Corresponds to the "Strings" type in the data model.
  }];
}

def IREEPyDM_TupleType : IREEPyDM_PrimitiveTypeDef<"Tuple"> {
  let mnemonic = "tuple";

  let summary = "Immutable sequence of boxed values";

  let description = [{
    Corresponds to the "Tuples" type in the data model.
  }];
  let extraClassDeclaration = [{
    /// Gets the type used to store elements in the backing list.
    /// For empty or boxed lists, this will be a generic ObjectType.
    /// For unboxed lists, it will be the uniform element type.
    Type getElementStorageType() const;
  }];
}

def IREEPyDM_TypeType : IREEPyDM_PrimitiveTypeDef<"Type"> {
  let mnemonic = "type";

  let summary = "Type associated with a value";

  let description = [{
    Corresponds to the Python `Type` class. It is considered a primitive because
    the data model cannot be represented without it.
  }];
}

//===----------------------------------------------------------------------===//
// Boxed objects
//===----------------------------------------------------------------------===//

def IREEPyDM_ObjectType : IREEPyDM_TypeDef<
    "Object", [
      DeclareTypeInterfaceMethods<PythonTypeInterface, ["isRefinable"]>
    ]> {
  let mnemonic = "object";

  let summary = "Core data type having an identity, type and value";

  let description = [{
    In terms of a typical Python runtime, objects are the primary data type.
    An object can represent every primitive and user defined type and value
    in the system. The act of converting a primitive to an object is called
    boxing, and doing so gives it an identity. Objects can be unboxed to
    specific primitive types.

    The system will function dynamically if specified completely in
    terms of untyped object types. Objects can be parameterized with a specific
    primitive type to support progressive typeification.
  }];

  let parameters = (ins
    "::mlir::iree_compiler::IREE::PYDM::PrimitiveType":$primitiveType
  );

  let builders = [
    TypeBuilder<(ins), [{
      return Base::get($_ctxt, nullptr);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static bool isGenericObjectType(Type t) {
      if (auto objectType = t.dyn_cast_or_null<ObjectType>())
        return !objectType.getPrimitiveType();
      return false;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Union type
//===----------------------------------------------------------------------===//

def IREEPyDM_UnionType : IREEPyDM_TypeDef<"Union"> {
  let mnemonic = "union";

  let summary = "Represents a union of types";

  let description = [{
    Unions show up naturally in Python data-flows and this type represents
    them. Note that it is not a "real" type in that it has not runtime
    realization. However, most ops can be parameterized in terms of unions, and
    it is generally the job of the compiler to do something sensible with them
    before lowering.
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$alternatives
  );

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Predicates and aggregate definitions
//===----------------------------------------------------------------------===//

def IREEPyDM_PrimitiveType : Type<CPred<
  "$_self.isa<::mlir::iree_compiler::IREE::PYDM::PrimitiveType>()">,
  "Python unboxed primitive type">;

def IREEPyDM_AnyValueType : AnyTypeOf<[
  IREEPyDM_ObjectType,
  IREEPyDM_PrimitiveType,
], "Python boxed or unboxed value">;

def IREEPyDM_AnyNumericType : AnyTypeOf<[
  IREEPyDM_BoolType,
  IREEPyDM_IntegerType,
  IREEPyDM_RealType,
], "Python numeric type">;

def IREEPyDM_GenericObjectType : Type<
    CPred<"::mlir::iree_compiler::IREE::PYDM::ObjectType::isGenericObjectType($_self)">,
    "generic object",
    "::mlir::iree_compiler::IREE::PYDM::ObjectType">,
    BuildableType<"$_builder.getType<::mlir::iree_compiler::IREE::PYDM::ObjectType>(nullptr)">;

// TODO: Upstream this. Missing from OpBase.td.
def IREEPyDM_FlatSymbolRefArrayAttr :
    TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = ?;
}

#endif // IREE_DIALECTS_DIALECT_PYDM_IR_PYDM_DIALECT_TD

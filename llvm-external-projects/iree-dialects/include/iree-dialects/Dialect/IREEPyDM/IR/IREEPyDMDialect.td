// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_DIALECT_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_DIALECT_TD

include "iree-dialects/Dialect/IREEPyDM/IR/IREEPyDMBase.td"

//===----------------------------------------------------------------------===//
// Unboxed Primitive Types
//===----------------------------------------------------------------------===//

// Declare a new primitive type.
// When adding a new one, update the PrimitiveType::classof method in
// IREEPyDMDialect.h.
class IREEPyDM_PrimitiveTypeDef<string name> :
    TypeDef<IREEPyDM_Dialect, name, /*traits=*/[],
    /*baseCppClass=*/"::mlir::iree_pydm::PrimitiveType"> {
}

def IREEPyDM_AnyPrimitiveType : Type<
    CPred<"$_self.isa<::mlir::iree_pydm::PrimitiveType>()">,
    "unboxed primitive type",
    "::mlir::iree_pydm::PrimitiveType">;

def IREEPyDM_BoolType : IREEPyDM_PrimitiveTypeDef<"Bool"> {
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

def IREEPyDM_IntegerType : IREEPyDM_PrimitiveTypeDef<"Integer"> {
  let mnemonic = "integer";

  let summary = "Type of integer values";

  let description = [{
    Represents the `numbers.Integral` type in the data model. At this abstract
    level, it should be considered conformant with the data model (i.e.
    unbounded). However, compiler flags will generally be used to interpret
    this type in a more bounded fashion (i32, i64, etc).
  }];
}

def IREEPyDM_ListType : IREEPyDM_PrimitiveTypeDef<"List"> {
  let mnemonic = "list";

  let summary = "Mutable sequence of boxed values";

  let description = [{
    Corresponds to the "Lists" type in the data model.
  }];
}

def IREEPyDM_NoneType : IREEPyDM_PrimitiveTypeDef<"None"> {
  let mnemonic = "none";

  let summary = "Type of the single 'None' value";

  let description = [{
    Represents the 'None' type in the standard type hierarchy.
  }];
}

def IREEPyDM_RealType : IREEPyDM_PrimitiveTypeDef<"Real"> {
  let mnemonic = "real";

  let summary = "Type of floating point values";

  let description = [{
    Represents the `numbers.Real` type in the data model. At this abstract
    level, it should be considered conformant with the data model (i.e.
    double precision). However, compiler flags will generally be used to
    interpret this type in a more bounded fashion (f32).
  }];
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

def IREEPyDM_ObjectType : IREEPyDM_TypeDef<"Object"> {
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
    "::mlir::iree_pydm::PrimitiveType":$primitiveType
  );

  let printer = [{
    $_printer << getMnemonic();
    if (getImpl()->primitiveType)
      $_printer << "<" << getImpl()->primitiveType << ">";
  }];

  let parser = [{
    if (parser.parseOptionalLess())
      return get($_ctxt, nullptr);

    Type t;
    if ($_parser.parseType(t))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    if (auto primitiveType = t.dyn_cast<PrimitiveType>())
      return get($_ctxt, primitiveType);
    else {
      $_parser.emitError(
          $_parser.getNameLoc(), "expected a primitive type");
      return Type();
    }
  }];

  let extraClassDeclaration = [{
    static bool isGenericObjectType(Type t) {
      if (auto objectType = t.dyn_cast_or_null<ObjectType>())
        return !objectType.getPrimitiveType();
      return false;
    }
  }];
}

//===----------------------------------------------------------------------===//
// Predicates and aggregate definitions
//===----------------------------------------------------------------------===//

def IREEPyDM_PrimitiveType : Type<CPred<
  "$_self.isa<::mlir::iree_pydm::PrimitiveType>()">,
  "Python unboxed primitive type">;

def IREEPyDM_AnyValueType : AnyTypeOf<[
  IREEPyDM_ObjectType,
  IREEPyDM_PrimitiveType,
], "Python boxed or unboxed value">;

def IREEPyDM_GenericObjectType : Type<
    CPred<"::mlir::iree_pydm::ObjectType::isGenericObjectType($_self)">,
    "generic object",
    "::mlir::iree_pydm::ObjectType">,
    BuildableType<"$_builder.getType<::mlir::iree_pydm::ObjectType>(nullptr)">;

// TODO: Upstream this. Missing from OpBase.td.
def IREEPyDM_FlatSymbolRefArrayAttr :
    TypedArrayAttrBase<FlatSymbolRefAttr, "flat symbol ref array attribute"> {
  let constBuilderCall = ?;
}

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_DIALECT_TD

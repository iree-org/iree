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

#ifndef IREE_DIALECT_IREE_BASE
#define IREE_DIALECT_IREE_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE base dialect used for types common across IREE subdialects.
//===----------------------------------------------------------------------===//

// TODO(b/143787186): rename when old dialects are removed.
def IREE_Dialect : Dialect {
  let name = "iree";
  let cppNamespace = "IREE";

  let summary = [{
    A dialect used for types common across IREE subdialects.
  }];
}

//===----------------------------------------------------------------------===//
// General types and helpers
//===----------------------------------------------------------------------===//

def IREE_Bool :
    AnyTypeOf<[I1, I8], "boolean-storing type (1 or 8 -bit integer)">;
def IREE_Element : AnyTypeOf<[AnyInteger, AnyFloat]>;
def IREE_MemRef : MemRefOf<[IREE_Element]>;

//===----------------------------------------------------------------------===//
// Attribute constraints
//===----------------------------------------------------------------------===//

def FuncTypeAttr : TypeAttrBase<"FunctionType", "function type">;

class IntegerAttrInRange<int min, int max> : AttrConstraint<
    CPred<"$_self.cast<IntegerAttr>().getInt() >= " # min # " && "
          "$_self.cast<IntegerAttr>().getInt() <= " # max>,
    "within the range [" # min # ", " # max # "] inclusive">;


class AliasedSymbolRefAttr : Attr<CPred<"$_self.isa<FlatSymbolRefAttr>()">,
                                        "symbol reference attribute"> {
  let storageType = [{ FlatSymbolRefAttr }];
  let returnType = [{ StringRef }];
  let constBuilderCall = "$_builder.getSymbolRefAttr($0)";
}

//===----------------------------------------------------------------------===//
// NamedTuple type constraint
//===----------------------------------------------------------------------===//

class NamedTupleElement<int thisOrdinal, string thisName, Type thisType> {
  int ordinal = thisOrdinal;
  string name = thisName;
  Type type = thisType;
}

// Check if all of the tuple elements are buildable.
class AreTupleElementsBuildable<list<NamedTupleElement> lst> {
  bit ret = !foldl(1, lst, lhs, y, !and(lhs, !ne(y.type.builderCall, "")));
}

// TODO(b/143188410): support ordered tuples in OpBase.td.
class NamedTupleOf<list<NamedTupleElement> elements, string descr> :
    Type<And<[
      IsTupleTypePred,
      CPred<"$_self.cast<TupleType>().size() == " # !size(elements)>,
      And<!foreach(element, elements,
          SubstLeaves<
              "$_self",
              "$_self.cast<TupleType>().getType(" # element.ordinal # ")",
              element.type.predicate>)>
    ]>, descr> {
  // Generate a builder call if all of the elements are buildable.
  let builderCall = !if(!eq(AreTupleElementsBuildable<elements>.ret, 0),
    "", "$_builder.getTupleType({" #
        StrJoin<!foreach(t, elements, t.type.builderCall)>.result # "})"
  );
}

//===----------------------------------------------------------------------===//
// ref_ptr types
//===----------------------------------------------------------------------===//

def AnyRefObject : DialectType<
    IREE_Dialect,
    CPred<"$_self.isa<IREE::RefObjectType>()">,
    "ref_object"> {
  let typeDescription = [{
    An iree::RefObject subclass.
  }];
}

def AnyRefPtr : DialectType<
    IREE_Dialect,
    CPred<"$_self.isa<IREE::RefPtrType>()">,
    "ref"> {
  let typeDescription = [{
    An iree::ref_ptr<T> of some type.
  }];
}

class RefPtrOf<Type type> :
    Type<And<[
      CPred<"$_self.isa<IREE::RefPtrType>()">,
      SubstLeaves<"$_self", "$_self.cast<IREE::RefPtrType>().getObjectType()",
                  type.predicate>
    ]>, "ref<" # type.description # ">"> {
  // Set the builder call if the base type has a builder call.
  string builderCall = !if(!empty(type.builderCall),
                           "", "RefPtrType::get(" # type.builderCall # ")");
}

def ByteBufferType : DialectType<
    IREE_Dialect,
    CPred<"$_self.isa<IREE::ByteBufferType>()">,
    "byte_buffer"> {
  let typeDescription = [{
    A constant buffer of mapped host memory.
  }];
}
def ByteBufferRef : RefPtrOf<ByteBufferType>;
def ByteBufferRefAttr : AliasedSymbolRefAttr;

def MutableByteBufferType : DialectType<
    IREE_Dialect,
    CPred<"$_self.isa<IREE::MutableByteBufferType>()">,
    "mutable_byte_buffer"> {
  let typeDescription = [{
    A buffer of read-write host memory.
  }];
}
def MutableByteBufferRef : RefPtrOf<MutableByteBufferType>;

//===----------------------------------------------------------------------===//
// Common traits
//===----------------------------------------------------------------------===//

// TODO(b/143184519): add descriptions to other types.
// Operations with this trait indicate that they may yield execution of the
// current fiber. Invokers of the op must be yield-safe and assume that there
// may be a significant delay between when the operation is invoked and when it
// completes.
//
// TODO(benvanik): implement yield semantics.
// Need to add a new call type and function attr for 'async', then can validate
// entire parts of the call tree are either async-compatible or async-hostile.
// Only vm.call_async callees may contain an operation with YieldPoint.
def YieldPoint : NativeOpTrait<"IREE::YieldPoint">;

// Denotes that an operation is potentially "unsafe" if used.
// Untrusted modules containing ops marked as unsafe will fail to verify at
// runtime if loaded for dynamic execution.
def Unsafe : NativeOpTrait<"IREE::Unsafe">;

#endif  // IREE_DIALECT_IREE_BASE

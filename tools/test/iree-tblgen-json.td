// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: iree-tblgen --gen-dialect-json %s \
// RUN:   -I %S/../../compiler/src \
// RUN:   -I %S/../../third_party/llvm-project/mlir/include \
// RUN:   -I %S/../../external/_main~_repo_rules~llvm-project/mlir/include \
// RUN:   | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "iree/compiler/Utils/DocMetadata.td"

//===----------------------------------------------------------------------===//
// Test Dialect Definition
//===----------------------------------------------------------------------===//

def TestDialect : Dialect {
  let name = "test";
  let cppNamespace = "::mlir::iree_compiler::test";
  let summary = "Test dialect for iree-tblgen JSON generation.";
  let description = [{
    This dialect is used to test the JSON documentation generation
    functionality of iree-tblgen, including support for docMetadata.
  }];
}

// CHECK: "dialects": {
// CHECK-NEXT: "test": {
// CHECK-NEXT: "cppClassName": "TestDialect"
// CHECK: "summary": "Test dialect for iree-tblgen JSON generation."

//===----------------------------------------------------------------------===//
// Base Classes with docMetadata support
//===----------------------------------------------------------------------===//

class Test_Op<string mnemonic, list<Trait> traits = []> :
    Op<TestDialect, mnemonic, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_OpDocMetadata docMetadata = ?;
}

class Test_TypeDef<string name, list<Trait> traits = []>
    : TypeDef<TestDialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_TypeDocMetadata docMetadata = ?;
}

class Test_AttrDef<string name, list<Trait> traits = []>
    : AttrDef<TestDialect, name, traits> {
  // Optional documentation metadata for rich doc generation.
  Util_AttrDocMetadata docMetadata = ?;
}

class Test_OpInterface<string name> : OpInterface<name> {
  // Optional documentation metadata for rich doc generation.
  Util_InterfaceDocMetadata docMetadata = ?;
}

//===----------------------------------------------------------------------===//
// Test Operation
//===----------------------------------------------------------------------===//

def Test_FooOpDocMetadata : Util_OpDocMetadata {
  let testFile = "test/foo_test.mlir";
  let exampleRefs = ["@basicExample", "@advancedExample"];
  let relatedOps = ["test.bar", "test.baz"];
  let tags = ["experimental", "test"];
}

def Test_FooOp : Test_Op<"foo", []> {
  let summary = "A test operation.";
  let description = [{
    This operation is used to test JSON generation
    with multiple lines of description text.
  }];

  let arguments = (ins
    AnyType:$input,
    I32Attr:$count
  );

  let results = (outs
    AnyType:$output
  );

  let docMetadata = Test_FooOpDocMetadata;
}

// CHECK: "name": "test.foo"
// CHECK-NEXT: "dialect": "test"
// CHECK-NEXT: "cppClassName": "FooOp"
// CHECK: "metadata": {
// CHECK: "testFile": "{{.+}}foo_test.mlir"
// CHECK: "exampleRefs": [
// CHECK-NEXT: "@basicExample"
// CHECK: "tags": [
// CHECK-NEXT: "experimental"

//===----------------------------------------------------------------------===//
// Test Type
//===----------------------------------------------------------------------===//

def Test_CustomTypeDocMetadata : Util_TypeDocMetadata {
  let category = "Test Types";
  let relatedTypes = ["test.other"];
  let testFile = "test/custom_type_test.mlir";
  let exampleRefs = ["@typeExample"];
  let tags = ["test"];
}

def Test_CustomType : Test_TypeDef<"Custom"> {
  let mnemonic = "custom";
  let summary = "A custom test type.";
  let description = [{
    This type is used to test type definition
    JSON generation with docMetadata support.
  }];

  let assemblyFormat = "";

  let docMetadata = Test_CustomTypeDocMetadata;
}

// CHECK: "name": "Test_CustomType"
// CHECK-NEXT: "dialect": "test"
// CHECK-NEXT: "cppClassName": "CustomType"
// CHECK: "metadata": {
// CHECK: "category": "Test Types"
// CHECK: "relatedTypes": [
// CHECK-NEXT: "test.other"
// CHECK: "testFile": "{{.+}}custom_type_test.mlir"
// CHECK: "exampleRefs": [
// CHECK-NEXT: "@typeExample"
// CHECK: "tags": [
// CHECK-NEXT: "test"

//===----------------------------------------------------------------------===//
// Test Attribute
//===----------------------------------------------------------------------===//

def Test_CustomAttrDocMetadata : Util_AttrDocMetadata {
  let category = "Test Attributes";
  let relatedAttrs = ["test.other_attr"];
  let testFile = "test/custom_attr_test.mlir";
  let exampleRefs = ["@attrExample"];
  let tags = ["test"];
}

def Test_CustomAttr : Test_AttrDef<"Custom"> {
  let mnemonic = "custom";
  let summary = "A custom test attribute.";
  let description = [{
    This attribute is used to test attribute definition
    JSON generation with docMetadata support.
  }];

  let parameters = (ins
    "int64_t":$value
  );

  let assemblyFormat = "`<` $value `>`";

  let docMetadata = Test_CustomAttrDocMetadata;
}

// CHECK: "name": "Test_CustomAttr"
// CHECK-NEXT: "dialect": "test"
// CHECK-NEXT: "cppClassName": "CustomAttr"
// CHECK: "metadata": {
// CHECK: "category": "Test Attributes"
// CHECK: "relatedAttrs": [
// CHECK-NEXT: "test.other_attr"
// CHECK: "testFile": "{{.+}}custom_attr_test.mlir"
// CHECK: "exampleRefs": [
// CHECK-NEXT: "@attrExample"
// CHECK: "tags": [
// CHECK-NEXT: "test"

//===----------------------------------------------------------------------===//
// Test Interface
//===----------------------------------------------------------------------===//

def Test_FooInterfaceDocMetadata : Util_InterfaceDocMetadata {
  let category = "Test Interfaces";
  let relatedInterfaces = ["test.bar_interface"];
  let testFile = "test/interface_test.mlir";
  let exampleRefs = ["@interfaceExample"];
  let tags = ["test"];
}

def Test_FooInterface : Test_OpInterface<"FooOpInterface"> {
  let description = [{
    This interface is used to test interface JSON generation
    with docMetadata support.
  }];

  let methods = [
    InterfaceMethod<
      "Get the foo value",
      "int64_t", "getFoo", (ins)
    >,
  ];

  let docMetadata = Test_FooInterfaceDocMetadata;
}

// CHECK: "name": "Test_FooInterface"
// CHECK-NEXT: "cppClassName": "FooOpInterface"
// CHECK: "metadata": {
// CHECK: "category": "Test Interfaces"
// CHECK: "relatedInterfaces": [
// CHECK-NEXT: "test.bar_interface"
// CHECK: "testFile": "{{.+}}interface_test.mlir"
// CHECK: "exampleRefs": [
// CHECK-NEXT: "@interfaceExample"
// CHECK: "tags": [
// CHECK-NEXT: "test"

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

// Structural ops such as 'module' and 'executable'.
// These are used to organize IREE IR into regions representing ops that act at
// the sequencer level (coarse control flow/scheduling) and ops that perform
// actual work (math/etc) on runtime execution backends.

#ifndef IREE_STRUCTURE_OPS
#define IREE_STRUCTURE_OPS

#ifndef IREE_OP_BASE
include "iree/compiler/IR/OpBase.td"
#endif  // IREE_OP_BASE

class IREE_StructureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

def IREE_MultiArchExecutableOp :
    IREE_StructureOp<"multi_arch_executable", [
      SingleBlockImplicitTerminator<"MultiArchExecutableEndOp">
    ]> {
  let arguments = (ins
    StrAttr:$sym_name,
    OptionalAttr<I32Attr>:$ordinal
  );

  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    StringRef getName() {
      return this->getOperation()->template getAttrOfType<StringAttr>(
          ::mlir::SymbolTable::getSymbolAttrName()).getValue();
    }

    Region& getBody() {
      return this->getOperation()->getRegion(0);
    }
    Block& getBlock() {
      return this->getOperation()->getRegion(0).front();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, StringRef name">,
  ];
}

def IREE_MultiArchExecutableEndOp :
    IREE_StructureOp<"_multi_arch_executable_end", [
      IREE_MultiArchExecutableOnly,
      Terminator
    ]> {
  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];
}

def IREE_ExecutableOp :
    IREE_StructureOp<"executable", [
      SingleBlockImplicitTerminator<"ExecutableEndOp">,
      NativeOpTrait<"SymbolTable">
    ]> {
  let arguments = (ins
    IREE_ExecutableFormatAttr:$format,
    OptionalAttr<I32Attr>:$ordinal
  );

  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    Region& getBody() {
      return this->getOperation()->getRegion(0);
    }
    Block& getBlock() {
      return this->getOperation()->getRegion(0).front();
    }
    ::mlir::ModuleOp getInnerModule() {
      return *getBlock().getOps<::mlir::ModuleOp>().begin();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<[{Builder *builder, OperationState &state,
                ExecutableFormat executable_format}]>,
  ];
}

def IREE_ExecutableEndOp :
    IREE_StructureOp<"_executable_end", [Terminator, IREE_ExecutableOnly]> {
  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];
}

#endif  // IREE_STRUCTURE_OPS

// Structural ops such as 'module' and 'executable'.
// These are used to organize IREE IR into regions representing ops that act at
// the sequencer level (coarse control flow/scheduling) and ops that perform
// actual work (math/etc) on runtime execution backends.

#ifdef IREE_STRUCTURE_OPS
#else
#define IREE_STRUCTURE_OPS

#ifdef IREE_OP_BASE
#else
include "compiler/IR/OpBase.td"
#endif  // IREE_OP_BASE

class IREE_StructureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

def IREE_ModuleOp :
    IREE_StructureOp<"module", [
      SingleBlockImplicitTerminator<"ModuleEndOp">,
      NativeOpTrait<"SymbolTable">
    ]> {
  let regions = (region SizedRegion<1>:$body);
  let extraClassDeclaration = [{
    Block& getBlock() {
      return this->getOperation()->getRegion(0).front();
    }
  }];

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<"Builder *, OperationState &state">];
}

def IREE_ModuleEndOp :
    IREE_StructureOp<"_module_end", [
      IREE_ModuleOnly,
      Terminator
    ]> {
  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];
}

def IREE_MultiArchExecutableOp :
    IREE_StructureOp<"multi_arch_executable", [
      // TODO(benvanik): make iree.module work and make this IREE_ModuleOnly.
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

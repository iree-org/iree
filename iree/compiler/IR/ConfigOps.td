// Ops used to declare configuration used by the IREE compiler.
// These allow inline config that follows along the IR they are associated with.
// Multiple config ops are allowed within a single scope to indicate that the
// parent IR node should be processed for multiple targets.

#ifdef IREE_CONFIG_OPS
#else
#define IREE_CONFIG_OPS

include "iree/compiler/IR/OpBase.td"

class IREE_ConfigOp<string mnemonic, list<OpTrait> traits = []> :
    Op<IREE_Dialect, mnemonic, traits> {
  let parser = [{ return parse$cppClass(parser, result); }];
  let printer = [{ print$cppClass(p, *this); }];
}

//===----------------------------------------------------------------------===//
// iree.executable configuration
//===----------------------------------------------------------------------===//

def IREE_ExecutableTargetConfigOp : IREE_ConfigOp<"target_config", [
  IREE_ExecutableOnly,
  SingleBlockImplicitTerminator<"ExecutableTargetConfigEndOp">
]> {
  let arguments = (ins
     StrAttr:$backend
  );

  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<"Builder *builder, OperationState &state, std::string backend">,
  ];
}

def IREE_ExecutableTargetConfigEndOp :
    IREE_ConfigOp<"_target_config_end", [Terminator, IREE_ExecutableTargetConfigOnly]> {
  let parser = [{ return parseNoIOOp(parser, result); }];
  let printer = [{ printNoIOOp(getOperation(), p); }];
}

#endif  // IREE_CONFIG_OPS

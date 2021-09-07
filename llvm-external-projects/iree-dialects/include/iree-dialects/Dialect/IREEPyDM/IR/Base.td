// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_BASE_TD
#define IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_BASE_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def IREEPyDM_Dialect : Dialect {
  let name = "iree_pydm";
  let summary = "Python data model as expressible for compilation to IREE";
  let description = [{
    Provides an implementation of the Python Data Model
    (https://docs.python.org/3/reference/datamodel.html) as adapted to run
    on the IREE VM.

    This dialect aims for correctness of the subset of the Python Data Model
    supported by IREE, with future work focused on completeness. Goals:
      - Provide a suitable set of types and ops for trivially parsing Python
        ASTs to this dialect, in a similar fashion as the CPython interpreter
        parses to bytecode.
      - Focused on embedded Python program extraction, where subsets of
        programs are compiled from a running Python instance for later
        hermetic execution.
      - Makes IR decisions conducive to progress typeification, enabling
        optimization benefits compared to fully untyped programs.
  }];
  let cppNamespace = "::mlir::iree_pydm";
  let hasConstantMaterializer = 1;
}

class IREEPyDM_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<IREEPyDM_Dialect, mnemonic, traits>;
class IREEPyDM_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<IREEPyDM_Dialect, mnemonic, !listconcat(traits, [NoSideEffect])>;
class IREEPyDM_TypeDef<string name, list<Trait> traits = []> : TypeDef<IREEPyDM_Dialect, name, traits>;

#endif // IREE_LLVM_EXTERNAL_PROJECTS_IREE_DIALECTS_DIALECT_IREEPYDM_IR_BASE_TD

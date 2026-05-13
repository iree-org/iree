// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-rocdl-prepare-for-spirv"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ROCDLPREPAREFORSPIRVPASS
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"

namespace {

// Remaps AMDGPU address spaces to SPIR-V address spaces.
//   AMDGPU 5 (private/stack) -> SPIR-V 0 (Function)
//   AMDGPU 0 (flat/generic)  -> SPIR-V 4 (Generic)
//   AMDGPU 4 (constant)      -> SPIR-V 2 (UniformConstant)
//   AMDGPU 1 (global)        -> 1 (no change)
//   AMDGPU 3 (shared/LDS)    -> 3 (no change)
static unsigned remapAddressSpace(unsigned as) {
  switch (as) {
  case 5:
    return 0; // private -> Function
  case 0:
    return 4; // flat/generic -> Generic
  case 4:
    return 2; // constant -> UniformConstant
  default:
    return as; // 1 (global) and 3 (shared) stay the same
  }
}

// Remaps address spaces in a type. Recursively handles pointer and array types.
static Type remapType(Type type) {
  if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type)) {
    unsigned newAS = remapAddressSpace(ptrType.getAddressSpace());
    if (newAS != ptrType.getAddressSpace()) {
      return LLVM::LLVMPointerType::get(type.getContext(), newAS);
    }
    return type;
  }
  if (auto arrayType = dyn_cast<LLVM::LLVMArrayType>(type)) {
    Type newElem = remapType(arrayType.getElementType());
    if (newElem != arrayType.getElementType()) {
      return LLVM::LLVMArrayType::get(newElem, arrayType.getNumElements());
    }
    return type;
  }
  if (auto funcType = dyn_cast<LLVM::LLVMFunctionType>(type)) {
    Type newRet = remapType(funcType.getReturnType());
    SmallVector<Type> newParams;
    bool changed = newRet != funcType.getReturnType();
    for (Type param : funcType.getParams()) {
      Type newParam = remapType(param);
      newParams.push_back(newParam);
      changed |= newParam != param;
    }
    if (changed) {
      return LLVM::LLVMFunctionType::get(newRet, newParams,
                                         funcType.isVarArg());
    }
    return type;
  }
  if (auto structType = dyn_cast<LLVM::LLVMStructType>(type)) {
    if (structType.isOpaque()) {
      return type;
    }
    SmallVector<Type> newBody;
    bool changed = false;
    for (Type elem : structType.getBody()) {
      Type newElem = remapType(elem);
      newBody.push_back(newElem);
      changed |= newElem != elem;
    }
    if (changed) {
      // TODO: Identified structs lose their name here. This is acceptable
      // for the current IR patterns but needs proper handling for recursive
      // struct types if they ever appear.
      return LLVM::LLVMStructType::getLiteral(type.getContext(), newBody,
                                              structType.isPacked());
    }
    return type;
  }
  return type;
}

// List of AMDGPU function attributes to remove for SPIR-V.
// Note: MLIR uses underscores in attribute names (target_cpu), while LLVM IR
// uses hyphens (target-cpu). We remove both forms.
static constexpr llvm::StringLiteral kAMDGPUAttrsToRemove[] = {
    "amdgpu-flat-work-group-size",
    "amdgpu_flat_work_group_size",
    "uniform-work-group-size",
    "uniform_work_group_size",
    "amdgpu-waves-per-eu",
    "amdgpu_waves_per_eu",
    "target-cpu",
    "target_cpu",
    "target-features",
    "target_features",
    // ROCDL-specific attributes set by ROCDLAnnotateKernelForTranslationPass.
    "rocdl.kernel",
    "rocdl.flat_work_group_size",
    "rocdl.reqd_work_group_size",
    "rocdl.max_flat_work_group_size",
};

static bool isOptimizationLevelFlag(StringRef arg) {
  return arg == "-O0" || arg == "-O1" || arg == "-O2" || arg == "-O3" ||
         arg == "-Os" || arg == "-Oz";
}

static std::string ensureO3InCmdline(StringRef cmdline) {
  SmallVector<StringRef> args;
  cmdline.split(args, '\0', /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  std::string result;
  bool hasOptLevel = false;
  for (StringRef arg : args) {
    if (isOptimizationLevelFlag(arg)) {
      if (hasOptLevel) {
        continue;
      }
      arg = "-O3";
      hasOptLevel = true;
    }
    result.append(arg.data(), arg.size());
    result.push_back('\0');
  }

  if (!hasOptLevel) {
    result.append("-O3", 3);
    result.push_back('\0');
  }

  return result;
}

struct ROCDLPrepareForSPIRVPass final
    : impl::ROCDLPrepareForSPIRVPassBase<ROCDLPrepareForSPIRVPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Set module triple and data layout for SPIR-V.
    moduleOp->setAttr(
        LLVM::LLVMDialect::getTargetTripleAttrName(),
        StringAttr::get(moduleOp.getContext(), "spirv64-amd-amdhsa"));
    // Data layout from llvm::Triple("spirv64-amd-amdhsa").computeDataLayout().
    // Must match exactly or MachineFunction::init() asserts.
    moduleOp->setAttr(
        LLVM::LLVMDialect::getDataLayoutAttrName(),
        StringAttr::get(
            moduleOp.getContext(),
            "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128"
            "-v192:256-v256:256-v512:512-v1024:1024-n32:64-S32-G1-P4-A0"));

    // Process all functions: remap calling conventions and remove AMDGPU attrs.
    moduleOp->walk([&](LLVM::LLVMFuncOp funcOp) {
      // Remap calling conventions.
      // After ROCDLAnnotateKernelForTranslationPass, kernel functions have
      // rocdl.kernel attribute instead of amdgpu_kernel CC.
      auto cc = funcOp.getCConv();
      if (cc == LLVM::CConv::AMDGPU_KERNEL || funcOp->hasAttr("rocdl.kernel")) {
        funcOp.setCConv(LLVM::CConv::SPIR_KERNEL);
      } else if (!funcOp.getName().starts_with("llvm.")) {
        // All non-intrinsic functions (definitions and declarations) get
        // spir_func. LLVM intrinsics (llvm.*) keep their default CC.
        funcOp.setCConv(LLVM::CConv::SPIR_FUNC);
      }

      // Remove AMDGPU attributes from the function's attribute dictionary.
      for (auto attrName : kAMDGPUAttrsToRemove) {
        funcOp->removeAttr(attrName);
      }

      // Also clean llvm_func_attrs if present.
      if (auto funcAttrs =
              funcOp->getAttrOfType<DictionaryAttr>("llvm_func_attrs")) {
        SmallVector<NamedAttribute> newAttrs;
        for (NamedAttribute attr : funcAttrs) {
          if (!llvm::is_contained(kAMDGPUAttrsToRemove, attr.getName())) {
            newAttrs.push_back(attr);
          }
        }
        if (newAttrs.size() != funcAttrs.size()) {
          if (newAttrs.empty()) {
            funcOp->removeAttr("llvm_func_attrs");
          } else {
            funcOp->setAttr(
                "llvm_func_attrs",
                DictionaryAttr::get(moduleOp.getContext(), newAttrs));
          }
        }
      }

      // Remove inreg attributes from all parameters. AMDGPU uses inreg for
      // argument preloading which is incompatible with SPIR-V.
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        funcOp.removeArgAttr(i, "llvm.inreg");
      }

      // Remap function signature types (pointer address spaces in
      // params/return).
      auto funcType = funcOp.getFunctionType();
      Type newRetType = remapType(funcType.getReturnType());
      SmallVector<Type> newParamTypes;
      bool sigChanged = newRetType != funcType.getReturnType();
      for (Type param : funcType.getParams()) {
        Type newParam = remapType(param);
        newParamTypes.push_back(newParam);
        sigChanged |= newParam != param;
      }
      if (sigChanged) {
        funcOp.setFunctionType(LLVM::LLVMFunctionType::get(
            newRetType, newParamTypes, funcType.isVarArg()));
        // Also update block argument types.
        if (!funcOp.isDeclaration()) {
          Block &entryBlock = funcOp.getBody().front();
          for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments())) {
            if (i < newParamTypes.size()) {
              arg.setType(newParamTypes[i]);
            }
          }
        }
      }
    });

    // Remove llvm.intr.assume calls. The SPIR-V backend's SPIRVEmitIntrinsics
    // crashes on llvm.assume with operand bundles (paramHasAttr out-of-bounds).
    SmallVector<LLVM::AssumeOp> assumeOps;
    moduleOp->walk([&](LLVM::AssumeOp op) { assumeOps.push_back(op); });
    for (auto op : assumeOps) {
      op.erase();
    }

    // Remap address spaces on all operations.
    moduleOp->walk([&](Operation *op) {
      // Skip function ops — they were already handled above.
      if (isa<LLVM::LLVMFuncOp>(op)) {
        return;
      }

      // Handle ops with explicit addr_space attributes first.
      if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(op)) {
        // The address space is encoded in the result pointer type.
        unsigned oldAS = allocaOp.getRes().getType().getAddressSpace();
        unsigned newAS = remapAddressSpace(oldAS);
        if (newAS != oldAS) {
          allocaOp.getRes().setType(
              LLVM::LLVMPointerType::get(op->getContext(), newAS));
        }
        // Also remap the addr_space attribute if present (discardable attr).
        if (auto addrSpaceAttr =
                allocaOp->getAttrOfType<IntegerAttr>("addr_space")) {
          unsigned attrAS = addrSpaceAttr.getInt();
          unsigned newAttrAS = remapAddressSpace(attrAS);
          if (newAttrAS != attrAS) {
            allocaOp->setAttr(
                "addr_space",
                IntegerAttr::get(addrSpaceAttr.getType(), newAttrAS));
          }
        }
        return;
      }
      if (auto globalOp = dyn_cast<LLVM::GlobalOp>(op)) {
        unsigned oldAS = globalOp.getAddrSpace();
        unsigned newAS = remapAddressSpace(oldAS);
        if (newAS != oldAS) {
          globalOp.setAddrSpace(newAS);
        }
        Type newGlobalType = remapType(globalOp.getGlobalType());
        if (newGlobalType != globalOp.getGlobalType()) {
          globalOp.setGlobalType(newGlobalType);
        }
        return;
      }

      // For all other ops, remap result types.
      for (auto [i, resultType] : llvm::enumerate(op->getResultTypes())) {
        Type newType = remapType(resultType);
        if (newType != resultType) {
          op->getResult(i).setType(newType);
        }
      }
    });

    // Embed @llvm.cmdline with "-O3" so comgr JIT compiles at -O3.
    // Without this, comgr defaults to -O0, causing massive register spilling.
    // See amd/comgr/src/comgr-compiler.cpp: extractSpirvFlags().
    // Must be at addrspace(1) (CrossWorkgroup) so the SPIR-V backend emits it
    // as a module-level global that survives the round-trip.
    auto i8Type = IntegerType::get(moduleOp.getContext(), 8);
    auto cmdlineGlobal = moduleOp.lookupSymbol<LLVM::GlobalOp>("llvm.cmdline");
    if (cmdlineGlobal) {
      auto valueAttr = dyn_cast_if_present<StringAttr>(
          cmdlineGlobal.getValueAttr());
      if (!valueAttr) {
        cmdlineGlobal.emitOpError()
            << "expected @llvm.cmdline to have a string initializer";
        return signalPassFailure();
      }
      std::string flags = ensureO3InCmdline(valueAttr.getValue());
      cmdlineGlobal.setValueAttr(StringAttr::get(moduleOp.getContext(), flags));
      cmdlineGlobal.setGlobalType(
          LLVM::LLVMArrayType::get(i8Type, flags.size()));
      cmdlineGlobal.setSection(".llvmcmd");
      cmdlineGlobal.setAlignment(1);
      cmdlineGlobal.setAddrSpace(1);
    } else {
      StringRef flags("-O3\0", 4);
      OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->end());
      auto globalOp = LLVM::GlobalOp::create(
          builder, moduleOp.getLoc(),
          LLVM::LLVMArrayType::get(i8Type, flags.size()),
          /*isConstant=*/true,
          LLVM::Linkage::Private, "llvm.cmdline", builder.getStringAttr(flags));
      globalOp.setSection(".llvmcmd");
      globalOp.setAlignment(1);
      globalOp.setAddrSpace(1);
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler

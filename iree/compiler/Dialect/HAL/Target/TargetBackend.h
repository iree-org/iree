// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_

#include <functional>
#include <string>
#include <vector>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Utils/DeviceSwitchBuilder.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Controls executable translation targets.
struct TargetOptions {
  // TODO(benvanik): multiple targets of the same type, etc.
  std::vector<std::string> targets;

  // TODO(benvanik): flags for debug/optimization/etc.
  // The intent is that we can have a global debug/-ON flag that then each
  // target backend can have tickle it's own flags in the right way. Right now
  // the best we can do is a coarse flag as to whether source maps should be
  // embedded, however we could be much better here on the TargetBackend
  // interface.
  void bindOptions(OptionsBinder &binder);
  using FromFlags = OptionsFromFlags<TargetOptions>;
};

// HAL executable target backend interface.
// Multiple backends can be registered and targeted during a single compilation.
// The flow->hal conversion process will use registered TargetBackend interfaces
// to query for scheduling parameters (such as workgroup size), allow for
// backend-specific scheduling logic (such as custom command buffer dispatch
// recording), and to setup the transformation pipeline.
//
// During each phase of lowering the executable may be duplicated based on the
// target configuration. For example, a single input `flow.executable` will map
// to at least one `hal.executable.variant` for each unique target backend
// configuration, and for each of those target backends can emit one or more
// `hal.executable.variant` containing the translated contents. Finally, each
// executable target will be serialized into one or more binary formats. The
// exact contents of the `hal.executable.variant` ops is left to the backends
// and can contain backend-specific nested IR and attributes.
//
// Hypothetical example (Vulkan+SPIR-V):
//   -> flow.executable @my_exe
//   [[-iree-hal-materialize-interfaces]]
//   -> hal.executable @my_exe
//      + hal.executable.variant @spirv-v1.1-mobile filter="spirv-v1.1-mobile*"
//          hal.executable.entry_point @my_entry
//          module { ... }
//      + hal.executable.variant @spirv-v1.1-desktop
//      filter="spirv-v1.1-desktop*"
//          hal.executable.entry_point @my_entry
//          module { ... }
//      + hal.executable.variant @spirv-v1.2-desktop
//      filter="spirv-v1.2-desktop*"
//          hal.executable.entry_point @my_entry
//          module { ... }
//   [[-iree-hal-translate-executables]]
//   -> hal.executable @my_exe
//      + hal.executable.variant @spirv-v1.1-mobile filter="spirv-v1.1-mobile*"
//          hal.executable.entry_point @my_entry_1
//          hal.executable.entry_point @my_entry_2
//          hal.executable.entry_point @my_entry_3
//          module { spv.module { ... } }
//      + hal.executable.variant @spirv-v1.1-desktop
//      filter="spirv-v1.1-desktop*"
//          hal.executable.entry_point @my_entry
//          module { spv.module { ... } }
//      + hal.executable.variant @spirv-v1.2-desktop
//      filter="spirv-v1.2-desktop*"
//          hal.executable.entry_point @my_entry
//          module { spv.module { ... } }
//   [[-iree-hal-link-executables]]
//   -> TODO(benvanik): linkage rules.
//   [[-iree-hal-serialize-executables]]
//   -> hal.executable @my_exe
//      + hal.executable.binary attributes { ... }
//          data blob...
//      + hal.executable.binary attributes { ... }
//          data blob...
//      + hal.executable.binary attributes { ... }
//          data blob...
//      + hal.executable.binary attributes { ... }
//          data blob...
class TargetBackend {
 public:
  virtual ~TargetBackend() = default;

  // Returns a name for the backend used to differentiate between other targets.
  virtual std::string name() const = 0;

  // Returns the name of the runtime device for this backend.
  // TODO(benvanik): remove this once we can properly specify targets.
  virtual std::string deviceID() const { return name(); }

  // Registers dependent dialects for the TargetBackend.
  // Mirrors the method on mlir::Pass of the same name. A TargetBackend is
  // expected to register the dialects it will create entities for (Operations,
  // Types, Attributes).
  virtual void getDependentDialects(DialectRegistry &registry) const {}

  // Returns the default device this backend targets.
  virtual IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const = 0;

  // Inserts passes used to translate the `hal.executable.variant` op contents.
  // The pass manager will be nested on `hal.executable` such that the pipeline
  // will only run on executable contents.
  //
  // Backend transformation passes must check that the source op they receive
  // is for them using the `target_backend` attribute. Backends may have
  // multiple source ops in the same executable to transform such as when
  // multiple target configurations are requested.
  //
  // For example, as input:
  //   hal.executable @some_executable {
  //     hal.interface @main_io {
  //       hal.interface.binding @arg0, set=0, binding=0, ...
  //       hal.interface.binding @arg1, set=0, binding=1, ...
  //     }
  //     hal.executable.variant @target, target="target-backend" {
  //       hal.executable.entry_point @main interface(@main_io) {
  //         ordinal = 0 : index
  //       }
  //       module { ... }
  //     }
  //   }
  //
  // As output:
  //   hal.executable @some_executable {
  //     hal.interface @main_io ...
  //     hal.executable.variant @target, target="target-backend" {
  //       hal.executable.entry_point @main ...
  //       module { spv.module { ... } }
  //     }
  //   }
  virtual void buildTranslationPassPipeline(OpPassManager &passManager) = 0;

  // Links compatible executables within the provided |moduleOp| together into
  // zero or more new linked executables. Implementations should move
  // executable contents (including interfaces, entry points, and functions)
  // into new executables and update any relevant references as they do so.
  //
  // Which executables to link together and how many new executables to produce
  // are left to implementations to determine. For example, an implementation
  // may choose to link all executables (even with different interfaces) into
  // a single combined executable, or it could choose to limit the number linked
  // together in order to shard binary size across multiple executables.
  //
  // The input |moduleOp| may contain executables containing multiple targets,
  // so implementations should check target backend filters against their own
  // `name()` prior to modifying them.
  //
  // Sample output structure:
  //   hal.executable @linked_executable {
  //     hal.interface @io_0 { ... }
  //     hal.interface @io_1 { ... }
  //     hal.executable.variant @target, target="target-backend" {
  //       hal.executable.entry_point @main_dispatch_0 attributes { ... }
  //       hal.executable.entry_point @main_dispatch_1 attributes { ... }
  //       hal.executable.entry_point @main_dispatch_2 attributes { ... }
  //       module {
  //         func @main_0(...) { ... }
  //         func @main_1(...) { ... }
  //         func @main_2(...) { ... }
  //       }
  //     }
  //   }
  //   // Other targets within executables are not modified
  //   hal.executable @main_dispatch_0 {
  //     hal.interface @io { ... }
  //     hal.executable.variant @other, target="other" {
  //       hal.executable.entry_point @main_dispatch_0 attributes { ... }
  //       module { ... }
  //     }
  //   }
  virtual LogicalResult linkExecutables(mlir::ModuleOp moduleOp) {
    return success();
  }

  // Serializes the given |variantOp| executable produced by this backend to one
  // or more binary byte buffer formats used for storage in the module file.
  // Implementations should insert `hal.executable.binary` ops for each format
  // (such as x64 and arm64 for compiled LLVM blobs, etc).
  //
  // If no serialization is provided then lowering the parent module into a
  // binary format (such as to the IREE VM) will fail.
  virtual LogicalResult serializeExecutable(
      IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
    llvm_unreachable("unimplemented serializeExecutable");
    return failure();
  }

 protected:
  // Links all executables for the current target found in |moduleOp| into
  // |linkedExecutableOp|. Functions will be cloned into |linkedModuleOp|.
  LogicalResult linkExecutablesInto(
      mlir::ModuleOp moduleOp,
      ArrayRef<IREE::HAL::ExecutableOp> sourceExecutableOps,
      IREE::HAL::ExecutableOp linkedExecutableOp,
      IREE::HAL::ExecutableVariantOp linkedTargetOp,
      std::function<Operation *(mlir::ModuleOp moduleOp)> getInnerModuleFn,
      OpBuilder &builder);
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_

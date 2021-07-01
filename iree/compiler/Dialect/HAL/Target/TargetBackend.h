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
#include "iree/compiler/Dialect/HAL/Utils/TypeUtils.h"
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
};

// Returns a TargetOptions struct initialized with the
// --iree-hal-target-* flags.
TargetOptions getTargetOptionsFromFlags();

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
//   [[-iree-hal-translate-executable-variants]]
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
  // Returns a generic host-like set of constraints.
  static BufferConstraintsAttr makeDefaultBufferConstraints(
      MLIRContext *context);

  virtual ~TargetBackend() = default;

  // Returns a name for the backend used to differentiate between other targets.
  virtual std::string name() const = 0;

  // Returns the name of the runtime device for this backend.
  // TODO(benvanik): remove this once we can properly specify targets.
  virtual std::string deviceID() const { return name(); }

  // Queries for compile-time known buffer constraints.
  // These should conservatively represent the min/max values even if the
  // backend may support others at runtime.
  virtual BufferConstraintsAttr queryBufferConstraints(MLIRContext *context);

  // Register dependent dialects for the TargetBackend.
  // Mirrors the method on mlir::Pass of the same name. A TargetBackend is
  // expected to register the dialects it will create entities for (Operations,
  // Types, Attributes) in |declareVariantOps|.
  virtual void getDependentDialects(DialectRegistry &registry) const {}

  // Creates an interface representing the bindings and push constants required
  // to dispatch the executable. Interfaces used across backends and executables
  // will be deduplicated to reduce code size and runtime overhead and being
  // consistent with the conventions used by the default extraction will ensure
  // that maximum reuse is possible.
  //
  // TODO(benvanik): document default interface layout, when defined.
  // TODO(benvanik): multiple interfaces.
  // virtual IREE::HAL::InterfaceOp extractInterface(
  //     IREE::Flow::ExecutableOp sourceOp);

  // Creates zero or more hal.executable.variant ops for the target backend.
  // The target op's inner module should be constructed with any attributes
  // the backends wants to carry along during transformation and will later be
  // filled in with the flow.executable's contents.
  //
  // A backend may decide to create multiple variants of an executable given
  // different parameters or target device requirements. For example, if the
  // |sourceOp| represents a reduction the backend may produce:
  //   my-backend-v1-reduce-init
  //   my-backend-v1-reduce-downsample-aligned
  //   my-backend-v1-reduce-downsample-unaligned
  //   my-backend-v1-reduce-final
  // The `recordDispatch` implementation can then switch between these binaries
  // as needed based on dispatch context.
  virtual void declareVariantOps(IREE::Flow::ExecutableOp sourceOp,
                                 IREE::HAL::ExecutableOp executableOp);

  // Captured state from the point at which a dispatch is to be recorded.
  struct DispatchState {
    // The original flow.dispatch op.
    // This may contain additional placement hints or options and can be used to
    // carry across per-dispatch backend-specific flags.
    IREE::Flow::DispatchOp dispatchOp;

    // SSA value of the hal.device the command buffer is from.
    Value device;

    // SSA value of the hal.command_buffer to record into.
    Value commandBuffer;

    // Executable being dispatched, with translated target ops nested as
    // `hal.executable.variant`. Backends can dispatch any of the available
    // target executables.
    IREE::HAL::ExecutableOp executableOp;

    // Entry point on the public executable API that is being dispatched.
    // Note that many entry points may exist within a single executable.
    IREE::HAL::ExecutableEntryPointOp entryPointOp;

    // ABI interface used by the |entryPointOp|.
    IREE::HAL::InterfaceOp interfaceOp;

    // SSA value of the loaded hal.executable_layout reference.
    Value executableLayout;

    // SSA values of the workgroup count of the dispatch. See `flow.dispatch`
    // for more information on how this is calculated.
    SmallVector<Value, 3> workgroupCount;

    // A base offset within the push constants array that all new push constants
    // must follow. Note that backend-specific push constants must have been
    // allocated during `extractInterface`.
    int basePushConstantOffset = 0;
  };

  // Records a dispatch to a command buffer given the dispatch state.
  // Push constants and bindings are already set and at minimum only a
  // `hal.command_buffer.dispatch` is required.
  //
  // If a backend wants to provide additional push constants it can push them
  // beginning at offset |dispatchState.basePushConstantOffset|. Note that the
  // push constants must have been declared by `extractInterface`.
  //
  // The provided |dispatchState.workgroupCount| can be used to access the
  // workgroup count values for dispatch as provided on the original
  // flow.dispatch op. These arbitrarily-ranked dimensions need to be adapted
  // into the target-dependent 3-D XYZ grid space.
  //
  // |dispatchState.operands| and |dispatchState.results| can be used to access
  // the buffers allocated in case additional command buffer operations are
  // needed. Note that any introduced scheduling dependency must be handled,
  // such as by inserting an  `hal.command_buffer.execution_barrier`.
  virtual LogicalResult recordDispatch(Location loc,
                                       DispatchState dispatchState,
                                       DeviceSwitchRewriter &switchRewriter);

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
  //       hal.executable.entry_point @main attributes {
  //         interface = @main_io,
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

  // Calculates the workgroup size (x, y, z). These are the dimension numbers
  // for a single workgroup.
  virtual std::array<Value, 3> calculateDispatchWorkgroupSize(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder);

  // Calculates the workgroup count (x, y, z) for dispatching to the given
  // |entryPointOp|. The provided N-dimensional |workload| is the total number
  // of invocations required as calculated by the generic workload logic
  // (basically, number of output elements in tensors).
  virtual std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, ValueRange workload,
      OpBuilder &builder);
  // Calculates the workgroup count (x, y, z) given the total N-dimensional
  // |workload| and specific |workgroupSize|.
  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, ValueRange workload,
      const std::array<Value, 3> &workgroupSize, OpBuilder &builder);
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_

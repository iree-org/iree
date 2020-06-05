// Copyright 2020 Google LLC
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
// to at least one `hal.executable.target` for each unique target backend
// configuration, and for each of those target backends can emit one or more
// `hal.executable.target` containing the translated contents. Finally, each
// executable target will be serialized into one or more binary formats. The
// exact contents of the `hal.executable.target` ops is left to the backends and
// can contain backend-specific nested IR and attributes.
//
// Hypothetical example (Vulkan+SPIR-V):
//   -> flow.executable @my_exe
//   [[-iree-hal-materialize-interfaces]]
//   -> hal.executable @my_exe
//      + hal.executable.target "vulkan-spirv-v1.1-mobile"
//          module { ... }
//      + hal.executable.target "vulkan-spirv-v1.1-desktop"
//          module { ... }
//      + hal.executable.target "vulkan-spirv-v1.2-desktop"
//          module { ... }
//   [[-iree-hal-translate-executables]]
//   -> hal.executable @my_exe
//      + hal.executable.target "vulkan-spirv-v1.1-mobile"
//          module { spv.module { ... } }
//      + hal.executable.target "vulkan-spirv-v1.1-desktop"
//          module { spv.module { ... } }
//      + hal.executable.target "vulkan-spirv-v1.2-desktop"
//          module { spv.module { ... } }
//   [[-iree-hal-link-executables]]
//   -> TODO(benvanik): linkage rules.
//   [[-iree-hal-serialize-executables]]
//   -> hal.executable @my_exe
//      + hal.executable.binary "vulkan-spirv-v1.1-mobile" "spirv-binary"
//          data blob...
//      + hal.executable.binary "vulkan-spirv-v1.1-mobile" "smolv-binary"
//          data blob...
//      + hal.executable.binary "vulkan-spirv-v1.1-desktop" "spirv-binary"
//          data blob...
//      + hal.executable.binary "vulkan-spirv-v1.2-desktop" "spirv-binary"
//          data blob...
class TargetBackend {
 public:
  // Returns true if the given |value| matches |pattern| (normal * and ? rules).
  // This accepts wildcards in the form of '*' and '?' for any delimited value.
  // '*' will match zero or more of any character and '?' will match exactly one
  // of any character.
  //
  // For example:
  // 'foo-*-bar' matches: 'foo-123-bar', 'foo-456-789-bar'
  // 'foo-10?' matches: 'foo-101', 'foo-102'
  static bool matchPattern(StringRef value, StringRef pattern);

  virtual ~TargetBackend() = default;

  // Returns the name of the backend as expected to be matched with a call to
  // matchPattern. For example, 'vulkan-v1.1' or 'vmla*'.
  virtual std::string name() const = 0;

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

  // Creates zero or more hal.executable.target ops for the target backend.
  // The target op will contain the flow.executable contents and any attributes
  // the backend wants to carry along during transformation.
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
  virtual void constructTargetOps(IREE::Flow::ExecutableOp sourceOp,
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
    // `hal.executable.target`. Backends can dispatch any of the available
    // target executables.
    IREE::HAL::ExecutableOp executableOp;

    // Entry point on the public executable API that is being dispatched.
    // Note that many entry points may exist within a single executable.
    IREE::HAL::ExecutableEntryPointOp entryPointOp;

    // SSA value of the loaded hal.executable reference.
    Value executable;

    // SSA value of the loaded hal.executable_layout reference.
    Value executableLayout;

    // SSA value of the total workload of the dispatch. See `flow.dispatch` for
    // more information on how this is calculated.
    Value workload;

    // A base offset within the push constants array that all new push constants
    // must follow. Note that backend-specific push constants must have been
    // allocated during `extractInterface`.
    int basePushConstantOffset = 0;

    // Dispatch operands in a form accessible as hal.buffer/hal.buffer_view.
    // Note that any introduced scheduling dependency (such as a write of an
    // operand/result prior to the dispatch) must be handled appropriately, such
    // as by inserting a `hal.command_buffer.barrier`.
    //
    // Operands are 1:1 the flow.dispatch operands, meaning that if there were
    // operands that were not tensor/buffer types they will be None.
    //
    // NOTE: some operands/results may alias (as indicated by the interface).
    ArrayRef<Optional<TensorRewriteAdaptor>> operands;

    // Dispatch results with allocated buffers.
    // Note that any introduced scheduling dependency (such as a write of an
    // operand/result prior to the dispatch) must be handled appropriately, such
    // as by inserting a `hal.command_buffer.barrier`.
    //
    // Results are 1:1 the flow.dispatch results, meaning that if there were
    // results that were not tensor/buffer types they will be None.
    //
    // NOTE: some operands/results may alias (as indicated by the interface).
    ArrayRef<Optional<TensorRewriteAdaptor>> results;
  };

  // Records a dispatch to a command buffer given the dispatch state.
  // Push constants and bindings are already set and at minimum only a
  // `hal.command_buffer.dispatch` is required.
  //
  // If a backend wants to provide additional push constants it can push them
  // beginning at offset |dispatchState.basePushConstantOffset|. Note that the
  // push constants must have been declared by `extractInterface`.
  //
  // The provided |dispatchState.workload| can be used to derive the workgroup
  // counts for dispatch using `calculateDispatchWorkgroupCounts` (or other
  // logic).
  //
  // |dispatchState.operands| and |dispatchState.results| can be used to access
  // the buffers allocated in case additional command buffer operations are
  // needed. Note that any introduced scheduling dependency must be handled,
  // such as by inserting an  `hal.command_buffer.execution_barrier`.
  virtual LogicalResult recordDispatch(Location loc,
                                       DispatchState dispatchState,
                                       DeviceSwitchBuilder &switchBuilder);

  // Inserts passes used to translate the `hal.executable.target` op contents.
  // The pass manager will be nested on `hal.executable` such that the pipeline
  // will only run on executable contents.
  //
  // Backend transformation passes must check that the source op they receive
  // is for them using the `target_backend` attribute. Backends may have
  // multiple source ops in the same executable to transform such as when
  // multiple target configurations are requested. Use the `matchPattern`
  // utility when comparing the target backend name.
  //
  // For example, as input:
  //   hal.executable @some_executable {
  //     hal.interface @main_io {
  //       hal.interface.binding @arg0, set=0, binding=0, ...
  //       hal.interface.binding @arg1, set=0, binding=1, ...
  //     }
  //     hal.executable.entry_point @main attributes {
  //       interface = @main_io,
  //       ordinal = 0 : i32,
  //       signature = (tensor<4xf32>) -> tensor<4xf32>
  //     }
  //     hal.executable.target "target-backend" {
  //       module { ... }
  //     }
  //   }
  //
  // As output:
  //   hal.executable @some_executable {
  //     hal.interface @main_io ...
  //     hal.executable.entry_point @main ...
  //     hal.executable.target "target-backend" {
  //       module { spv.module { ... } }
  //     }
  //   }
  // TODO(benvanik): migrate this to the dynamic pipeline pass infra when it
  // exists. This will likely change to be a function that registers handlers
  // for target-specific name, attributes, etc. For now the executable target
  // is passed to allow snooping.
  virtual void buildTranslationPassPipeline(
      IREE::HAL::ExecutableTargetOp targetOp, OpPassManager &passManager) = 0;

  // TODO(benvanik): define linkage rules.
  // Major thing to figure out here is how to rewrite the executable references.
  // We may want to move executable selection into the hal.device.switch of the
  // dispatches so that they can be more easily replaced per-target.
  virtual LogicalResult linkExecutables(mlir::ModuleOp moduleOp) {
    return success();
  }

  // Serializes the given |targetOp| executable produced by this backend to one
  // or more binary byte buffer formats used for storage in the module file.
  // Implementations should insert `hal.executable.binary` ops for each format
  // (such as x64 and arm64 for compiled LLVM blobs, etc).
  //
  // If no serialization is provided then lowering the parent module into a
  // binary format (such as to the IREE VM) will fail.
  virtual LogicalResult serializeExecutable(
      IREE::HAL::ExecutableTargetOp targetOp, OpBuilder &executableBuilder) {
    llvm_unreachable("unimplemented serializeExecutable");
    return failure();
  }

 protected:
  // Calculates the workgroup size (x, y, z). Tese are the dimension numbers for
  // a single workgroup.
  virtual std::array<Value, 3> calculateDispatchWorkgroupSize(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
      OpBuilder &builder);

  // Calculates the workgroup count (x, y, z) for dispatching to the given
  // |entryPointOp|. The provided |workload| is the total number of invocations
  // required as calculated by the generic workload logic (basically, number of
  // output elements in tensors).
  virtual std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, IREE::HAL::ExecutableOp executableOp,
      IREE::HAL::ExecutableEntryPointOp entryPointOp, Value workload,
      OpBuilder &builder);
  // Calculates the workgroup count (x, y, z) given the total |workload| and
  // specific |workgroupSize|.
  std::array<Value, 3> calculateDispatchWorkgroupCount(
      Location loc, Value workload, const std::array<Value, 3> &workgroupSize,
      OpBuilder &builder);
};

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_

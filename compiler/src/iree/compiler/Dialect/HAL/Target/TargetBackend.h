// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

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
//          hal.executable.export @my_entry
//          module { ... }
//      + hal.executable.variant @spirv-v1.1-desktop
//      filter="spirv-v1.1-desktop*"
//          hal.executable.export @my_entry
//          module { ... }
//      + hal.executable.variant @spirv-v1.2-desktop
//      filter="spirv-v1.2-desktop*"
//          hal.executable.export @my_entry
//          module { ... }
//   [[-iree-hal-translate-all-executables]]
//   -> hal.executable @my_exe
//      + hal.executable.variant @spirv-v1.1-mobile filter="spirv-v1.1-mobile*"
//          hal.executable.export @my_entry_1
//          hal.executable.export @my_entry_2
//          hal.executable.export @my_entry_3
//          module { spirv.module { ... } }
//      + hal.executable.variant @spirv-v1.1-desktop
//      filter="spirv-v1.1-desktop*"
//          hal.executable.export @my_entry
//          module { spirv.module { ... } }
//      + hal.executable.variant @spirv-v1.2-desktop
//      filter="spirv-v1.2-desktop*"
//          hal.executable.export @my_entry
//          module { spirv.module { ... } }
//   [[-iree-hal-link-all-executables]]
//   -> TODO(benvanik): linkage rules.
//   [[-iree-hal-serialize-all-executables]]
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

  // Returns the ID of a DeviceTarget that can execute files produced by this
  // backend. This is used to support the `--iree-hal-target-backends=` flag
  // and will be removed in the future.
  virtual std::string getLegacyDefaultDeviceID() const = 0;

  // Appends zero or more executable targets for a device with the given
  // ID and configuration using flags/options that control target defaults.
  virtual void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const = 0;

  // Appends zero or more executable targets for a device with the given
  // ID and configuration that represents the hosting machine.
  virtual void getHostExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const {}

  // Provides a list of supported element types by the target compilation
  // backend. Supported element types vary by target-dependent information like
  // specific ISA and compilation modes available on an executable target
  // configuration. Compilation backends should use their information directly
  // but higher layers of the compiler may use the coarse queries provided here
  // to select targets.
  class SupportedTypes {
  public:
    void addScalarType(Type type) { scalarTypes.insert(type); }
    bool supportsScalarType(Type type) const {
      return scalarTypes.contains(type);
    }

    void addElementType(Type type) { elementTypes.insert(type); }
    bool supportsElementType(Type type) const {
      return elementTypes.contains(type);
    }

    bool supportsType(Type type) const {
      if (auto tensorType = llvm::dyn_cast<TensorType>(type)) {
        return supportsElementType(tensorType.getElementType());
      } else {
        return supportsScalarType(type);
      }
    }

  private:
    llvm::DenseSet<Type> scalarTypes;
    llvm::DenseSet<Type> elementTypes;
  };

  // Returns a set of types that are supported as compilation input based on
  // the target configuration.
  virtual SupportedTypes getSupportedTypes(MLIRContext *context) const;

  // Registers dependent dialects for the TargetBackend.
  // Mirrors the method on mlir::Pass of the same name. A TargetBackend is
  // expected to register the dialects it will create entities for (Operations,
  // Types, Attributes).
  virtual void getDependentDialects(DialectRegistry &registry) const {}

  // Inserts passes used to configure the `hal.executable.variant` op contents
  // for translation. The pass manager will be nested on `hal.executable` such
  // that the pipeline will only run on executable contents.
  //
  // The primary purpose of this pipeline is to preprocess then annotate all
  // `hal.executable.variant` ops with the information necessary for
  // translation. This can include specifying the set of required and/or
  // irrelevant target features, allowing for an additional deduplication step
  // when the only difference between two variants is a set of irrelevant
  // features. As a result, this pipeline is optional.
  //
  // The expected input to this pipeline might look like:
  //   hal.executable @some_executable {
  //     hal.interface @main_io {
  //       hal.interface.binding @arg0, set=0, binding=0, ...
  //       hal.interface.binding @arg1, set=0, binding=1, ...
  //     }
  //     hal.executable.variant @target, target="target-backend" {
  //       hal.executable.export @main interface(@main_io) {
  //         ordinal = 0 : index
  //       }
  //       module {
  //         func.func @main ...
  //       }
  //     }
  //   }
  //
  // As output, structurally the variant should be very similar:
  //   hal.executable @some_executable {
  //     hal.interface @main_io {
  //       hal.interface.binding @arg0, set=0, binding=0, ...
  //       hal.interface.binding @arg1, set=0, binding=1, ...
  //     }
  //     hal.executable.variant @target, target="target-backend" {
  //       hal.executable.export @main interface(@main_io)
  //         {attrs = #target_specific_translation_attr<...>} {
  //         ordinal = 0 : index
  //       }
  //       module {
  //         func.func @main ...
  //       }
  //     }
  //   }
  virtual void
  buildConfigurationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                                 OpPassManager &passManager) {}

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
  //       hal.executable.export @main interface(@main_io) {
  //         ordinal = 0 : index
  //       }
  //       module { ... (annotated for translation) }
  //     }
  //   }
  //
  // As output:
  //   hal.executable @some_executable {
  //     hal.interface @main_io ...
  //     hal.executable.variant @target, target="target-backend" {
  //       hal.executable.export @main ...
  //       module { spirv.module { ... } }
  //     }
  //   }
  virtual void
  buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr targetAttr,
                               OpPassManager &passManager) = 0;

  // Inserts passes used to link `hal.executable.variant` ops together.
  // The pass manager will be nested on the parent module of `hal.executable`
  // ops and the pipeline will need to find relevant variant ops itself.
  //
  // Implementations should clone executable contents (including interfaces,
  // entry points, and functions) into new executables and update any relevant
  // references as they do so.
  //
  // Which executable variants to link together and how many new executables to
  // produce are left to implementations to determine. For example, an
  // implementation may choose to link all executables (even with different
  // interfaces) into a single combined executable, or it could choose to limit
  // the number linked together in order to shard binary size across multiple
  // executables.
  //
  // For example, as input:
  //   hal.executable @some_executable_0 {
  //     hal.interface...
  //     hal.executable.variant @target_a, target="target-backend" {
  //       module { ... }
  //     }
  //   }
  //   hal.executable @some_executable_1 {
  //     hal.interface...
  //     hal.executable.variant @target_b, target="target-backend" {
  //       module { ... }
  //     }
  //     hal.executable.variant @target_c, target="other-backend" {
  //       module { ... }
  //     }
  //   }
  //
  // As output:
  //   hal.executable @some_executable_1 {  // untouched, not relevant
  //     hal.interface...
  //     hal.executable.variant @target_c, target="other-backend" {
  //       module { ... }
  //     }
  //   }
  //   hal.executable @some_executable_linked {
  //     hal.interface...
  //     hal.executable.variant @target_a, target="target-backend" {
  //       module { ... }
  //     }
  //     hal.executable.variant @target_b, target="target-backend" {
  //       module { ... }
  //     }
  //   }
  virtual void buildLinkingPassPipeline(OpPassManager &passManager) {}

  struct SerializationOptions {
    // Debug level for serialization (0-3).
    int debugLevel;
    // File name prefix used when creating scratch files.
    // This contains the module and executable name in canonical form.
    // Example: some_module_executable_43
    std::string dumpBaseName;
    // Optional path to write temporary/intermediate files into.
    std::string dumpIntermediatesPath;
    // Optional path to write serialized binary results into.
    std::string dumpBinariesPath;
  };

  // Serializes the given |variantOp| executable produced by this backend to one
  // or more binary byte buffer formats used for storage in the module file.
  // Implementations should insert `hal.executable.binary` ops for each format
  // (such as x64 and arm64 for compiled LLVM blobs, etc).
  //
  // If no serialization is provided then lowering the parent module into a
  // binary format (such as to the IREE VM) will fail.
  virtual LogicalResult
  serializeExecutable(const SerializationOptions &options,
                      IREE::HAL::ExecutableVariantOp variantOp,
                      OpBuilder &executableBuilder) {
    assert(false && "unimplemented serializeExecutable");
    return failure();
  }
};

// Returns a sorted uniqued set of target backends used in the executable.
SmallVector<std::string>
gatherExecutableTargetNames(IREE::HAL::ExecutableOp executableOp);

// Returns a sorted uniqued set of target backends used in the entire module.
SmallVector<std::string> gatherExecutableTargetNames(mlir::ModuleOp moduleOp);

// Dumps binary data to a file formed by joining the given path components:
//   `path/baseName_suffix[extension]`
void dumpDataToPath(StringRef path, StringRef baseName, StringRef suffix,
                    StringRef extension, StringRef data);
template <typename T>
void dumpDataToPath(StringRef path, StringRef baseName, StringRef suffix,
                    StringRef extension, ArrayRef<T> data) {
  dumpDataToPath(path, baseName, suffix, extension,
                 StringRef(reinterpret_cast<const char *>(data.data()),
                           data.size() * sizeof(T)));
}

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_DIALECT_HAL_TARGET_TARGETBACKEND_H_

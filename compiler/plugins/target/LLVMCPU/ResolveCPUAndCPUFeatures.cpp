// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/ResolveCPUAndCPUFeatures.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/X86TargetParser.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

bool resolveHostCPUAndCPUFeatures(const llvm::Triple &triple, std::string &cpu,
                                  std::string &cpuFeatures) {
  if (cpu != "host" && cpuFeatures != "host") {
    return true;
  }
  if ((!cpu.empty() && cpu != "host") ||
      (!cpuFeatures.empty() && cpuFeatures != "host")) {
    llvm::errs()
        << "error: If either cpu or CpuFeatures is `host`, the other must "
           "be either also `host` or the default value\n";
    return false;
  }
  cpu = llvm::sys::getHostCPUName();
  llvm::SubtargetFeatures features;
  for (auto &feature : llvm::sys::getHostCPUFeatures()) {
    features.AddFeature(feature.first(), feature.second);
  }
  cpuFeatures = features.getString();
  return true;
}

bool logUnspecifiedCPUOrCPUFeatures(
    const llvm::Triple &triple, std::string &cpu, std::string &cpuFeatures,
    std::string_view loggingUnspecifiedTargetCPU) {
  if (!cpu.empty() || !cpuFeatures.empty()) {
    return true;
  }
  if (loggingUnspecifiedTargetCPU.empty()) {
    return false;
  }
  if (triple.isX86()) {
    // On X86, one usually talks in terms of CPU, not individual CPU features.
    llvm::errs() << R"MSG(
error: Please pass --iree-llvmcpu-target-cpu (or alternatively, --iree-llvmcpu-target-cpu-features, but this message is tailored to the current target architecture, x86, where --iree-llvmcpu-target-cpu is used more often).

Examples:

    --iree-llvmcpu-target-cpu=host
        Target the host CPU. The generated code will have optimal performance on the host CPU but will crash on other CPUs not supporting the same CPU features.

    --iree-llvmcpu-target-cpu=generic
        Target a generic CPU for the target architecture. The generated code will have poor performance unless CPU features are also specified by --iree-llvmcpu-target-cpu-features.

    --iree-llvmcpu-target-cpu=somecpu
        Target the specified `somecpu` (see accepted values below). The generated code will have optimal performance on the specified CPU, but will crash on older CPUs and will have suboptimal performance on newer CPUs.

Accepted CPU values:
)MSG";
    llvm::SmallVector<llvm::StringRef> allAcceptedCpus;
    llvm::X86::fillValidCPUArchList(allAcceptedCpus, /*Only64Bit=*/true);
    llvm::interleaveComma(allAcceptedCpus, llvm::errs());
    llvm::errs() << "\n";
  } else {
    // Outside of X86, one usually talks in terms of CPU features, but people
    // should also be aware of the possibility of specifying a CPU.
    llvm::errs() << R"MSG(
error: Please pass --iree-llvmcpu-target-cpu-features and/or -iree-llvmcpu-target-cpu. Outside of x86, --iree-llvmcpu-target-cpu-features is the preferred way of enabling CPU features.

Examples:

    --iree-llvmcpu-target-cpu-features=+somefeature1,+somefeature2,...
        Enable the comma-separated list of CPU features (each prefixed with a + sign).

    --iree-llvmcpu-target-cpu=host
        Target the host CPU. The generated code will have optimal performance on the host CPU but will crash on other CPUs not supporting the same CPU features.

    --iree-llvmcpu-target-cpu=generic
        Target a generic CPU for the target architecture. The generated code will have poor performance unless CPU features are also specified by --iree-llvmcpu-target-cpu-features.

    --iree-llvmcpu-target-cpu=somecpu
        WARNING: outside of x86, this flag may not actually populate CPU features. It may only serve to select the instruction scheduling model.
)MSG";
  }

  if (loggingUnspecifiedTargetCPU == "fatal") {
    // I know we shouldn't do that in library code. But this is a non-default
    // flag, and it's going to be very useful to check that we got all instances
    // fixed before enabling the warning by default.
    exit(EXIT_FAILURE);
  }
  return false;
}

bool resolveCPUFeaturesForCPU(const llvm::Triple &triple, std::string &cpu,
                              std::string &cpuFeatures) {
  if (!cpuFeatures.empty()) {
    // Explicitly specified CPU features: not overriding.
    return true;
  }
  if (cpu.empty() || cpu == "generic" ||
      llvm::StringRef(cpu).starts_with("generic-")) {
    // Implicitly (default) or explicitly specified generic CPU: no features.
    // Logging (on unspecified CPU) is handled elsewhere, no need for it here.
    return true;
  }
  llvm::SubtargetFeatures targetCpuFeatures(cpuFeatures);
  auto addCpuFeatures = [&](const auto &getFeaturesForCPU,
                            auto &cpuFeatureList) {
    getFeaturesForCPU(cpu, cpuFeatureList, false);
    for (const auto &feature : cpuFeatureList) {
      targetCpuFeatures.AddFeature(feature);
    }
  };
  if (triple.isX86()) {
    llvm::SmallVector<llvm::StringRef> cpuFeatureList;
    addCpuFeatures(llvm::X86::getFeaturesForCPU, cpuFeatureList);
  } else if (triple.isRISCV64()) {
    llvm::SmallVector<std::string> cpuFeatureList;
    addCpuFeatures(llvm::RISCV::getFeaturesForCPU, cpuFeatureList);
  } else {
    llvm::errs() << "error: Resolution of target CPU to target CPU features is "
                    "not implemented on this target architecture. Pass "
                    "explicit CPU features instead of a CPU "
                    "on this architecture, or implement that.\n";
    return false;
  }
  cpuFeatures = targetCpuFeatures.getString();
  return true;
}

bool tweakCPUFeatures(const llvm::Triple &triple, std::string &cpu,
                      std::string &cpuFeatures) {
  if (triple.isAArch64()) {
    llvm::SubtargetFeatures targetCpuFeatures(cpuFeatures);
    // x18 is platform-reserved per the Aarch64 procedure call specification.
    targetCpuFeatures.AddFeature("reserve-x18", true);
    cpuFeatures = targetCpuFeatures.getString();
  }
  return true;
}

} // namespace

bool resolveCPUAndCPUFeatures(const llvm::Triple &triple, std::string &cpu,
                              std::string &cpuFeatures,
                              std::string_view loggingUnspecifiedTargetCPU) {
  bool success = true; // No early-return on error.
  success &= logUnspecifiedCPUOrCPUFeatures(triple, cpu, cpuFeatures,
                                            loggingUnspecifiedTargetCPU);
  success &= resolveHostCPUAndCPUFeatures(triple, cpu, cpuFeatures);
  success &= resolveCPUFeaturesForCPU(triple, cpu, cpuFeatures);
  success &= tweakCPUFeatures(triple, cpu, cpuFeatures);
  return success;
}

} // namespace mlir::iree_compiler::IREE::HAL

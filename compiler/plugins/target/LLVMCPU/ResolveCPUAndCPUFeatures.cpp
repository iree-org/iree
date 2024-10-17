// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/ResolveCPUAndCPUFeatures.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/RISCVTargetParser.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/TargetParser/X86TargetParser.h"

namespace mlir::iree_compiler::IREE::HAL {

namespace {

ResolveCPUAndCPUFeaturesStatus
resolveHostCPUAndCPUFeatures(std::string &cpu, std::string &cpuFeatures) {
  if (cpu != "host" && cpuFeatures != "host") {
    return ResolveCPUAndCPUFeaturesStatus::OK;
  }
  if ((!cpu.empty() && cpu != "host") ||
      (!cpuFeatures.empty() && cpuFeatures != "host")) {
    return ResolveCPUAndCPUFeaturesStatus::InconsistentHost;
  }
  cpu = llvm::sys::getHostCPUName();
  llvm::SubtargetFeatures features;
  for (auto &feature : llvm::sys::getHostCPUFeatures()) {
    features.AddFeature(feature.first(), feature.second);
  }
  cpuFeatures = features.getString();
  return ResolveCPUAndCPUFeaturesStatus::OK;
}

ResolveCPUAndCPUFeaturesStatus
resolveCPUFeaturesForCPU(const llvm::Triple &triple, std::string &cpu,
                         std::string &cpuFeatures) {
  if (!cpuFeatures.empty()) {
    // Explicitly specified CPU features: not overriding.
    return ResolveCPUAndCPUFeaturesStatus::OK;
  }
  if (cpu.empty() || cpu == "generic" ||
      llvm::StringRef(cpu).starts_with("generic-")) {
    // Implicitly (default) or explicitly specified generic CPU: no features.
    // Logging (on unspecified CPU) was already handled, no need for it here.
    return ResolveCPUAndCPUFeaturesStatus::OK;
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
    return ResolveCPUAndCPUFeaturesStatus::UnimplementedMapping;
  }
  cpuFeatures = targetCpuFeatures.getString();
  return ResolveCPUAndCPUFeaturesStatus::OK;
}

void tweakCPUFeatures(const llvm::Triple &triple, std::string &cpu,
                      std::string &cpuFeatures) {
  if (triple.isAArch64()) {
    llvm::SubtargetFeatures targetCpuFeatures(cpuFeatures);
    // Helper to add a feature if not already present. This check matters as
    // we check for equality of features to tell whether to generate the error
    // about implicitly targeting a generic CPU.
    auto addFeature = [&](const char *feature) {
      if (!targetCpuFeatures.hasFlag(std::string("+") + feature)) {
        targetCpuFeatures.AddFeature(feature, true);
      }
    };
    // x18 is platform-reserved per the Aarch64 procedure call specification.
    addFeature("reserve-x18");
    cpuFeatures = targetCpuFeatures.getString();
  }
}

std::string getImplicitGenericFallbackMessage(std::string_view triple_str) {
  llvm::Triple triple(triple_str);
  std::string msg = R"MSG(
Defaulting to targeting a generic CPU for the target architecture will result in poor performance. Please specify a target CPU and/or a target CPU feature set. If it is intended to target a generic CPU, specify "generic" as the CPU.

This can be done in two ways:
1. With command-line flags:
    --iree-llvmcpu-target-cpu=...
    --iree-llvmcpu-target-cpu-features=...
2. Within the IR:
    #hal.executable.target< ... , cpu="...", cpu_features="...">

In the rest of this message, these fields are referred to as just `cpu` and `cpu_features`.

Examples:

    cpu=generic
        Target a generic CPU of the target architecture. The generated code will have poor performance, but will run on any CPU.

    cpu=host
        Target the host CPU. The generated code will have optimal performance on the host CPU but will crash on other CPUs not supporting the same CPU features.

    cpu="name"
        Target a specific CPU. This is mostly used on x86. The accepted values are the same as in Clang command lines.)MSG";
  if (triple.isX86()) {
    msg += R"MSG(
        List of accepted x86 CPUs: )MSG";
    llvm::SmallVector<llvm::StringRef> allAcceptedCpus;
    llvm::X86::fillValidCPUArchList(allAcceptedCpus, /*Only64Bit=*/true);
    llvm::raw_string_ostream s(msg);
    llvm::interleaveComma(allAcceptedCpus, s);
    msg += "\n";
  } else {
    msg += R"MSG(
        CAVEAT: Outside of x86, this may only set the instruction scheduling model but may not enable CPU features. That's why when targeting non-x86 CPUs, it is usually preferred to pass cpu_features, see below.
)MSG";
  }
  msg += R"MSG(
    cpu_features="+feature1,..."
        Target a CPU supporting the comma-separated of (+-prefixed) features. The accepted values are the same as in Clang command lines.
)MSG";
  if (triple.isAArch64()) {
    msg += R"MSG(
        Example: cpu_features="+dotprod,+i8mm,+bf16
)MSG";
  }
  if (triple.isRISCV()) {
    msg += R"MSG(
        Example: cpu_features="+m,+a,+f,+d,+c
)MSG";
  }
  return msg;
}

} // namespace

ResolveCPUAndCPUFeaturesStatus
resolveCPUAndCPUFeatures(std::string_view triple_str, std::string &cpu,
                         std::string &cpuFeatures) {
  llvm::Triple triple(triple_str);
  // No early-return on error status. The caller may treat these errors as
  // non-fatal and will carry on with whichever `cpu` and `cpuFeatures` we
  // produce.
  auto status1 = resolveHostCPUAndCPUFeatures(cpu, cpuFeatures);
  auto status2 = resolveCPUFeaturesForCPU(triple, cpu, cpuFeatures);
  tweakCPUFeatures(triple, cpu, cpuFeatures);

  std::string defaultTweakedCpu;
  std::string defaultTweakedCpuFeatures;
  tweakCPUFeatures(triple, defaultTweakedCpu, defaultTweakedCpuFeatures);

  auto status3 =
      (cpu == defaultTweakedCpu && cpuFeatures == defaultTweakedCpuFeatures)
          ? ResolveCPUAndCPUFeaturesStatus::ImplicitGenericFallback
          : ResolveCPUAndCPUFeaturesStatus::OK;

  // Helper to return the first non-OK status.
  auto combine = [](ResolveCPUAndCPUFeaturesStatus a,
                    ResolveCPUAndCPUFeaturesStatus b) {
    return a == ResolveCPUAndCPUFeaturesStatus::OK ? b : a;
  };
  return combine(combine(status1, status2), status3);
}

std::string getMessage(ResolveCPUAndCPUFeaturesStatus status,
                       std::string_view triple_str) {
  switch (status) {
  case ResolveCPUAndCPUFeaturesStatus::ImplicitGenericFallback:
    return getImplicitGenericFallbackMessage(triple_str);
  case ResolveCPUAndCPUFeaturesStatus::InconsistentHost:
    return "If either CPU or CPU-features is `host`, the other must "
           "be either also `host` or the default value.\n";
  case ResolveCPUAndCPUFeaturesStatus::UnimplementedMapping:
    return "Resolution of CPU to CPU-features is not implemented on this "
           "target architecture. Pass explicit "
           "CPU-features, or implement the missing mapping.\n";
  default:
    assert(false);
    return "";
  }
}

} // namespace mlir::iree_compiler::IREE::HAL

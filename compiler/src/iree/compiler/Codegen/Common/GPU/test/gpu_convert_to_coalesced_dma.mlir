// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-convert-to-coalesced-dma))" %s --split-input-file | FileCheck %s

// CHECK-LABEL: module

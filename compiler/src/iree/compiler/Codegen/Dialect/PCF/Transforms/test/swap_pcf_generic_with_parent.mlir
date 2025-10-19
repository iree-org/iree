// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-convert-forall-to-loops)" --split-input-file | FileCheck %s

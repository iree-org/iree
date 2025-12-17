# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_compiler_register_plugin(
  PLUGIN_ID
    input_shardy
  TARGET
    ::registration
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRPass
    iree::compiler::PluginAPI
    iree::compiler::plugins::input::Shardy::InputConversion
    SdyDialect
    SdyDialectRegister
  PUBLIC
)

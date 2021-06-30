# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LIBYAML_ROOT "${IREE_ROOT_DIR}/third_party/libyaml/")

external_cc_library(
  PACKAGE
    yaml
  NAME
    yaml
  ROOT
    ${LIBYAML_ROOT}
  INCLUDES
    "${LIBYAML_ROOT}/include"
  SRCS
    "src/api.c"
    "src/dumper.c"
    "src/emitter.c"
    "src/loader.c"
    "src/parser.c"
    "src/reader.c"
    "src/scanner.c"
    "src/writer.c"
    "src/yaml_private.h"
  HDRS
    "include/yaml.h"
  COPTS
    "-D_CRT_SECURE_NO_WARNINGS"
    "-DYAML_DECLARE_STATIC"
    "-DYAML_VERSION_STRING=\"0.2.5\""
    "-DYAML_VERSION_MAJOR=0"
    "-DYAML_VERSION_MINOR=2"
    "-DYAML_VERSION_PATCH=5"
  INCLUDES
    "${LIBYAML_ROOT}/include"
  PUBLIC
)

if(MSVC)
  target_compile_options(yaml_yaml PRIVATE /wd4996)
endif()

# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# This is an example showing a basic makefile that links in the IREE runtime by
# way of the unified static library. It's recommended that IREE is added as a
# subproject and cmake is used to add the dependencies (as in the CMakeLists.txt
# in this directory) but when using other build systems this is easier to adapt.
#
# Configure the runtime:
#   cmake -GNinja -B ../iree-build-runtime/ . \
#       -DCMAKE_BUILD_TYPE=MinSizeRel \
#       -DIREE_SIZE_OPTIMIZED=ON
# Build the runtime:
#   cmake --build ../iree-build-runtime/ --target iree_runtime_unified
# Make this binary:
#   make custom-module-run-min RUNTIME_BUILD_DIR=../iree-build-runtime/
#
# Note that if IREE_SIZE_OPTIMIZED is used to build the runtime then the
# -DNDEBUG and -DIREE_STATUS_MODE=0 are required on any binaries using it.

RUNTIME_SRC_DIR ?= ../../runtime/src/
RUNTIME_BUILD_DIR ?= ../../../iree-build/

SRC_FILES := module.h module.cc main.c
INCLUDE_DIRS := ${RUNTIME_SRC_DIR}
INCLUDE_FLAGS := $(addprefix -I,${INCLUDE_DIRS})
LIBRARY_DIRS := \
		${RUNTIME_BUILD_DIR}/build_tools/third_party/flatcc/ \
		${RUNTIME_BUILD_DIR}/runtime/src/iree/runtime/
LINK_LIBRARIES := \
    iree_runtime_unified \
		flatcc_parsing
LIBRARY_FLAGS := $(addprefix -L,${LIBRARY_DIRS}) $(addprefix -l,${LINK_LIBRARIES})
CXX_FLAGS := -flto ${INCLUDE_FLAGS} ${LIBRARY_FLAGS}
MIN_FLAGS := \
		-s \
		-Os \
		-DNDEBUG \
		-DIREE_STATUS_MODE=0

all: custom-module-run custom-module-run-min
clean:
	rm -f custom-module-run custom-module-run-min

custom-module-run: ${SRC_FILES}
	${CXX} ${SRC_FILES} ${CXX_FLAGS} -o $@
custom-module-run-min: ${SRC_FILES}
	${CXX} ${SRC_FILES} ${CXX_FLAGS} ${MIN_FLAGS} -o $@

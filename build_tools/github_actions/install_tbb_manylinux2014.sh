#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# The version of tbb installed on manylinux2014 is too old to support the
# parallel STL libraries on the installed GCC9-based toolchain. Further,
# Intel *broke* compatibility starting in 2021 for GCC<=10.
# To make matters worse, the prior 2020 versions did not have cmake or
# install support.
# Shame on you Intel.
# See: https://community.intel.com/t5/Intel-oneAPI-Threading-Building/tbb-task-has-not-been-declared/m-p/1254418
# Since this is unlikely to be helpful outside of the old centos systems
# that manylinux2014 is based on (newer ones are based on Debian),
# we just tailor this specifically for docker images of that distro.

# You can test this with either an official manylinux2014 docker image or
# our special one (which is really only special in that it includes bazel):
# docker run --rm -it -v $(pwd):/work stellaraccident/manylinux2014_x86_64-bazel-4.2.2:latest /bin/bash

set -e

mkdir -p /tmp/libtbb_build
cd /tmp/libtbb_build
curl -o tbbsrc.tgz -L https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2020.3.tar.gz
tar xzf tbbsrc.tgz
cd oneTBB-*/

echo "****** BUILDING TBB ******"
make -j$(nproc)
cp -R include/* /usr/include
cp build/*_release/* /usr/lib64
echo "prefix=/usr
exec_prefix=${prefix}
libdir=${exec_prefix}/lib64
includedir=${prefix}/include

Name: Threading Building Blocks
Description: Intel's parallelism library for C++
URL: http://www.threadingbuildingblocks.org/
Version:
Libs: -ltbb
Cflags:
" > /usr/lib64/pkgconfig/tbb.pc

echo "****** DONE BUILDING TBB ******"

cd /
rm -Rf /tmp/libtbb_build

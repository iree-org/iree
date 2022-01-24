# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bazel macros for running lit tests."""

load(":lit_test.bzl", "lit_test", "lit_test_suite")

def iree_lit_test(
        name,
        cfg = "//:lit.cfg.py",
        tools = None,
        env = None,
        **kwargs):
    """A thin wrapper around lit_test with some opinionated settings.

    See the base lit_test for more details on argument meanings.

    Args:
      name: name for the test.
      cfg: string. lit config file.
      tools: label_list. tools that should be included on the PATH.
        llvm-symbolizer is added by default.
      env: string_dict. Environment variables available to the test at runtime.
        FILECHECK_OPTS=--enable-var-scope is added if FILECHECK_OPTS is not
        already set.
      **kwargs: additional keyword args to forward to the underyling lit_test.
    """

    tools = tools or []
    env = env or {}

    # Always include llvm-symbolizer so we get useful stack traces. Maybe it
    # would be better to force everyone to do this explicitly, but since
    # forgetting wouldn't cause the test to fail, only make debugging harder
    # when it does, I think better to hardcode it here.
    llvm_symbolizer = "@llvm-project//llvm:llvm-symbolizer"
    if llvm_symbolizer not in tools:
        tools.append(llvm_symbolizer)

    filecheck_env_var = "FILECHECK_OPTS"
    if filecheck_env_var not in env:
        env[filecheck_env_var] = "--enable-var-scope"

    lit_test(
        name = name,
        cfg = cfg,
        tools = tools,
        env = env,
        **kwargs
    )

def iree_lit_test_suite(
        name,
        cfg = "//:lit.cfg.py",
        tools = None,
        env = None,
        **kwargs):
    """A thin wrapper around lit_test_suite with some opinionated settings.

    See the base lit_test for more details on argument meanings.

    Args:
      name: name for the test suite.
      cfg: string. lit config file.
      tools: label_list. tools that should be included on the PATH.
        llvm-symbolizer is added by default.
      env: string_dict. Environment variables available to the test at runtime.
        FILECHECK_OPTS=--enable-var-scope is added if FILECHECK_OPTS is not
        already set.
      **kwargs: additional keyword args to forward to the underyling
        lit_test_suite.
    """
    tools = tools or []
    env = env or {}

    # Always include llvm-symbolizer so we get useful stack traces. Maybe it
    # would be better to force everyone to do this explicitly, but since
    # forgetting wouldn't cause the test to fail, only make debugging harder
    # when it does, I think better to hardcode it here.
    llvm_symbolizer = "@llvm-project//llvm:llvm-symbolizer"
    if llvm_symbolizer not in tools:
        tools.append(llvm_symbolizer)

    filecheck_env_var = "FILECHECK_OPTS"
    if filecheck_env_var not in env:
        env[filecheck_env_var] = "--enable-var-scope"

    lit_test_suite(
        name = name,
        cfg = cfg,
        tools = tools,
        env = env,
        **kwargs
    )

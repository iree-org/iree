# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Embeds data files into a C module."""

def clean_dep(dep):
    """Returns an absolute Bazel path to 'dep'.

    This is necessary when calling these functions from another workspace.
    """
    return str(Label(dep))

def iree_c_embed_data(
        name,
        srcs,
        c_file_output,
        h_file_output,
        testonly = False,
        strip_prefix = None,
        flatten = False,
        identifier = None,
        generator = clean_dep("//build_tools/embed_data:iree-c-embed-data"),
        **kwargs):
    """Embeds 'srcs' into a C module.

    Generates a header like:
        #if __cplusplus
        extern "C" {
        #endif // __cplusplus
        struct iree_file_toc_t {
          const char* name;             // the file's original name
          const char* data;             // beginning of the file
          size_t size;                  // length of the file
        };
        #if __cplusplus
        }
        #endif // __cplusplus

        #if __cplusplus
        extern "C" {
        #endif // __cplusplus
        const struct iree_file_toc_t* this_rule_name__create();
        #if __cplusplus
        }
        #endif // __cplusplus

    The 'this_rule_name()' function will return an array of iree_file_toc_t
    structs terminated by one that has NULL 'name' and 'data' fields.
    The 'data' field always has an extra null terminator at the end (which
    is not included in the size).

    Args:
      name: The rule name, which will also be the identifier of the generated
        code symbol.
      srcs: List of files to embed.
      c_file_output: The C implementation file to output.
      h_file_output: The H header file to output.
      testonly: If True, only testonly targets can depend on this target.
      strip_prefix: Strips this verbatim prefix from filenames (in the TOC).
      flatten: Removes all directory components from filenames (in the TOC).
      identifier: The identifier to use in generated names (defaults to name).
      generator: tool to use generate the embed data files.
      **kwargs: Args to pass to the cc_library.
    """
    generator_location = "$(location %s)" % generator
    if identifier == None:
        identifier = name
    flags = "--output_header='$(location %s)' --output_impl='$(location %s)'" % (
        h_file_output,
        c_file_output,
    )
    flags += " --identifier='%s'" % (identifier,)
    if strip_prefix != None:
        flags += " --strip_prefix='%s'" % (strip_prefix,)
    if flatten:
        flags += " --flatten"

    native.genrule(
        name = name + "__generator",
        srcs = srcs,
        outs = [
            c_file_output,
            h_file_output,
        ],
        tools = [generator],
        cmd = "%s $(SRCS) %s" % (generator_location, flags),
        testonly = testonly,
    )
    native.cc_library(
        name = name,
        hdrs = [h_file_output],
        srcs = [c_file_output],
        testonly = testonly,
        **kwargs
    )

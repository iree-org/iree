# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Macro for building e2e tests from a single source with multiple flags."""

load("//bindings/python:build_defs.oss.bzl", "iree_py_test")

def _normalize_dictionary(dictionary):
    """Wraps every value of dictionary in a list if it isn't one already."""
    for key, value in dictionary.items():
        if type(value) != type([]):
            dictionary[key] = [value]
    return dictionary

def _dictionary_product(dictionary):
    """Returns a named cartesian product of dictionary's values."""

    # Converts {'a': [1, 2], 'b': [3, 4]} into
    # [{'a': 1, 'b': 3}, {'a': 1, 'b': 4}, {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
    product = [[]]
    for values in dictionary.values():
        # Iteratively grow the elements of the product.
        product = [element + [value] for element in product for value in values]
    dicts = [{k: v for k, v in zip(dictionary, element)} for element in product]
    return dicts

def iree_e2e_cartesian_product_test_suite(
        name,
        srcs,
        main,
        flags_to_values,
        failing_configurations = None,
        tags = None,
        data = None,
        deps = None,
        size = None,
        python_version = "PY3",
        **kwargs):
    """Creates a test for each configuration and bundles a succeeding and failing test suite.

    Computes the cartesian product of `flags_to_values` and then creates a test
    for each element of that product. Tests specified in
    `failing_configurations` are bundled into a test suite suffixed with
    "_failing" and tagged to be excluded from CI and wildcard builds. All other
    tests are bundled into a suite with the same name as the macro.

    For example, given the following values

        flags_to_values = {
            "use_external_weights": True,
            "model": [
                "ResNet50",
                "MobileBert",
            ],
            "target_backends": [
                "tf",
                "iree_vmla"
                "iree_vulkan",
            ]
        }
        failing_configurations = [
            {
                "model": "MobileBert",
                "target_backends": "iree_vulkan",
            },
            {
                "model": "ResNet50",
            },
        ]

    the following passing and failing configurations would be generated:
        # Passing
        {use_exernal_weights: True, model: MobileBert, target_backends: tf}
        {use_exernal_weights: True, model: MobileBert, target_backends: iree_vmla}

        # Failing
        {use_exernal_weights: True, model: ResNet50,   target_backends: tf}
        {use_exernal_weights: True, model: ResNet50,   target_backends: iree_vmla}
        {use_exernal_weights: True, model: ResNet50,   target_backends: iree_vulkan}
        {use_exernal_weights: True, model: MobileBert, target_backends: iree_vulkan}

    Args:
      name:
        name of the generated passing test suite. If failing_configurations
        is not `None` then a test suite named name_failing will also be
        generated.
      srcs:
        src files for iree_py_test
      main:
        main file for iree_py_test
      failing_configurations:
        an iterable of dictionaries specifying which flag values the test is
        failing for. If a flag name is present in `flags_to_values`, but not
        present in `failing_configurations`, then all of the values in
        `flags_to_values[flag_name]` are included. (See `ResNet50` in the
        example above).
      flags_to_values:
        a dictionary of strings (flag names) to lists (of values for the flags)
        to take a cartesian product of.
      tags:
        tags to apply to the test. Note that as in standard test suites, manual
        is treated specially and will also apply to the test suite itself.
      data:
        external data for iree_py_test.
      deps:
        test dependencies for iree_py_test.
      size:
        size of the tests for iree_py_test.
      python_version:
        the python version to run the tests with. Uses python3 by default.
      **kwargs:
        any additional arguments that will be passed to the underlying tests and
        test_suite.
    """

    # Normalize flags_to_values to always have lists as its values.
    # e.g. {use_external_data: True} -> {use_external_data: [True]}
    flags_to_values = _normalize_dictionary(flags_to_values)

    all_flag_configurations = _dictionary_product(flags_to_values)

    failing_flag_configurations = []
    if failing_configurations != None:
        for failing_configuration in failing_configurations:
            failing_configuration = _normalize_dictionary(failing_configuration)

            # If a flag isn't specified in the failing configuration, assume it
            # is failing for all values of that flag.
            for key, values in flags_to_values.items():
                if key not in failing_configuration:
                    failing_configuration[key] = values

            failing_flag_configurations.extend(
                _dictionary_product(failing_configuration),
            )

    tests = []
    for flags in all_flag_configurations:
        # Check if this is a failing configuration.
        failing = flags in failing_flag_configurations

        # Append "_failing" to name if this is a failing configuration.
        test_name = name if not failing else name + "_failing"
        test_name = [test_name]
        for k, v in flags.items():
            # Only include the flag's value in the test name if it's not always
            # the same.
            if len(flags_to_values[k]) > 1:
                test_name.append(k)
                test_name.append(str(v))
        test_name = "__".join(test_name)
        tests.append(test_name)

        args = ["--{}={}".format(k, v) for k, v in flags.items()]

        py_test_tags = []
        if "target_backends" in flags:
            # TODO(GH-2175): Simplify this after backend names are standardized.
            # "iree_<driver>" --> "<driver>"
            backend = flags["target_backends"]
            if len(backend.split(",")) > 1:
                fail("Expected only one target backend but got '{}'".format(
                    backend,
                ))

            driver = backend.replace("iree_", "")
            if driver == "llvmjit":
                driver = "llvm"
            py_test_tags += ["driver={}".format(driver)]
            if tags != None:  # `is` is not supported.
                py_test_tags += tags

        # Add additional tags if this is a failing configuration.
        if failing:
            py_test_tags += [
                "failing",  # Only used for test_suite filtering below.
                "manual",
                "nokokoro",
                "notap",
            ]

        iree_py_test(
            name = test_name,
            main = main,
            srcs = srcs,
            args = args,
            data = data,
            deps = deps,
            size = size,
            tags = py_test_tags,
            python_version = python_version,
            **kwargs
        )

    native.test_suite(
        name = name,
        tests = tests,
        # Add "-failing" to exclude tests in `tests` that have the "failing"
        # tag.
        tags = tags + ["-failing"],
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )

    if failing_configurations != None:
        native.test_suite(
            name = name + "_failing",
            tests = tests,
            # Add "+failing" to only include tests in `tests` that have the
            # "failing" tag.
            tags = tags + ["+failing"],
            # If there are kwargs that need to be passed here which only apply
            # to the generated tests and not to test_suite, they should be
            # extracted into separate named arguments.
            **kwargs
        )

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

load("@iree//build_tools/bazel:deep_copy.bzl", "deep_copy")

def get_driver(backend):
    # TODO(#2175): Simplify this after backend names are standardized.
    driver = backend.replace("iree_", "")  # "iree_<driver>" --> "<driver>"
    return driver

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
        matrix,
        failing_configurations = None,
        tags = None,
        data = None,
        deps = None,
        size = None,
        python_version = "PY3",
        **kwargs):
    """Creates a test for each configuration and bundles a succeeding and failing test suite.

    Computes the cartesian product of `matrix` and then creates a test for each
    element of that product. Tests specified in `failing_configurations` are
    bundled into a test suite suffixed with "_failing" and tagged to be excluded
    from CI and wildcard builds. All other tests are bundled into a suite with
    the same name as the macro.

    For example, given the following values

        matrix = {
            "src": "external_model_test.py"
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
                "model": ["ResNet50"],
            },
        ]

    the following passing and failing configurations would be generated:
        # Passing
        {src: external_model_test.py, use_exernal_weights: True, model: MobileBert, target_backends: tf}
        {src: external_model_test.py, use_exernal_weights: True, model: MobileBert, target_backends: iree_vmla}

        # Failing
        {src: external_model_test.py, use_exernal_weights: True, model: ResNet50,   target_backends: tf}
        {src: external_model_test.py, use_exernal_weights: True, model: ResNet50,   target_backends: iree_vmla}
        {src: external_model_test.py, use_exernal_weights: True, model: ResNet50,   target_backends: iree_vulkan}
        {src: external_model_test.py, use_exernal_weights: True, model: MobileBert, target_backends: iree_vulkan}

    Args:
      name:
        name of the generated passing test suite. If failing_configurations
        is not `None` then a test suite named name_failing will also be
        generated.
      failing_configurations:
        an iterable of dictionaries specifying which matrix values the test is
        failing for. If a key is present in `matrix`, but not present in an
        entry of `failing_configurations`, then all of the values in
        `matrix[flag_name]` are included. (See `ResNet50` in the example above).
      matrix:
        a dictionary of strings to lists to take a cartesian product of. Values
        that are not lists are normalized to single-element lists. `src` is
        required and is extracted as the sole source file and main for the
        underlying python test. `target_backends` must be specified. All other
        keys are passed as command line flags to the test.
      tags:
        tags to apply to the test. Note that as in standard test suites, manual
        is treated specially and will also apply to the test suite itself.
      data:
        external data for py_test.
      deps:
        test dependencies for py_test.
      size:
        size of the tests for py_test.
      python_version:
        the python version to run the tests with. Uses python3 by default.
      **kwargs:
        any additional arguments that will be passed to the underlying tests and
        test_suite.
    """
    if not "target_backends" in matrix:
        fail("`target_backends` must be a key in `matrix`.")
    if not "src" in matrix:
        fail("`src` must be a key in `matrix`.")

    # Bazel will implictly mutate this variable's state if it is not copied.
    # This allows failing configurations to be reused in BUILD files.
    failing_configurations = deep_copy(failing_configurations)

    # Normalize matrix to always have lists as its values.
    # e.g. {use_external_data: True} -> {use_external_data: [True]}
    matrix = _normalize_dictionary(matrix)

    all_matrix_configurations = _dictionary_product(matrix)

    failing_matrix_configurations = []
    if failing_configurations != None:
        for failing_configuration in failing_configurations:
            failing_configuration = _normalize_dictionary(failing_configuration)

            for key in failing_configuration:
                if key not in matrix:
                    fail("Encountered unexpected key \"{}\" ".format(key) +
                         "in a failing configuration. Expected one of " +
                         "{}.".format(list(matrix.keys())))

            # If a flag isn't specified in the failing configuration, assume it
            # is failing for all values of that flag.
            for key, values in matrix.items():
                if key not in failing_configuration:
                    failing_configuration[key] = values

            failing_matrix_configurations.extend(
                _dictionary_product(failing_configuration),
            )

    tests = []
    for flags in all_matrix_configurations:
        # Check if this is a failing configuration.
        failing = flags in failing_matrix_configurations

        # These keys that are required and we have extra logic around.
        target_backend = flags.pop("target_backends")
        src = flags.pop("src")

        if len(target_backend.split(",")) > 1:
            fail("Multiple target backends cannot be specified at once, but " +
                 "got `{}`".format(flags["target_backends"]))

        # Append "_failing" to name if this is a failing configuration.
        test_name = name if not failing else name + "_failing"
        test_name = [test_name]

        # Append the meaningful part of the source file name
        if len(matrix["src"]) > 1:
            src_name = src
            if src_name.endswith(".py"):
                src_name = src_name[:-len(".py")]
            if src_name.endswith("_test"):
                src_name = src_name[:-len("_test")]
            test_name.append(src_name)

        # Append the target backend
        if len(matrix["target_backends"]) > 1:
            test_name.append(target_backend)

        # For all other flags, append their key and value if the value isn't
        # always the same.
        for k, v in flags.items():
            if len(matrix[k]) > 1:
                test_name.append(k)
                test_name.append(str(v))
        test_name = "__".join(test_name)
        tests.append(test_name)

        # Need to add back target_backends, since we pulled it out above.
        args = ["--target_backends={}".format(target_backend)] + [
            "--{}={}".format(k, v)
            for k, v in flags.items()
        ]
        py_test_tags = ["driver={}".format(get_driver(target_backend))]
        if tags != None:  # `is` is not supported.
            py_test_tags += tags

        # Add additional tags if this is a failing configuration.
        if failing:
            py_test_tags += [
                "failing",
                "manual",
                "nokokoro",
                "notap",
            ]

        native.py_test(
            name = test_name,
            main = src,
            srcs = [src],
            args = args,
            data = data,
            deps = deps,
            size = size,
            tags = py_test_tags,
            python_version = python_version,
            **kwargs
        )

    if tags == None:
        tags = []

    if len(all_matrix_configurations) > 1:
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
            tags = tags + [
                "+failing",
                "manual",
                "nokokoro",
                "notap",
            ],
            # If there are kwargs that need to be passed here which only apply
            # to the generated tests and not to test_suite, they should be
            # extracted into separate named arguments.
            **kwargs
        )

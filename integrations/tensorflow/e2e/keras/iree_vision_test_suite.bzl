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

"""Macro for building e2e keras vision model tests."""

load("//bindings/python:build_defs.oss.bzl", "iree_py_test")
load("@bazel_skylib//lib:new_sets.bzl", "sets")

def iree_vision_test_suite(
        name,
        models,
        datasets,
        backends,
        reference_backend,
        failing_configurations = None,
        external_weights = None,
        tags = None,
        deps = None,
        size = "large",
        python_version = "PY3",
        **kwargs):
    """Creates a test for each configuration and bundles a succeeding and failing test suite.

    Creates one test per dataset, backend, and model. Tests indicated in
    `failing_configurations` are bundled into a suite suffixed with "_failing"
    tagged to be excluded from CI and wildcard builds. All other tests are
    bundled into a suite with the same name as the macro.

    Args:
      name:
        name of the generated passing test suite. If failing_configurations is
        not `None` then a test suite named name_failing will also be generated.
      models:
        an iterable of model names to generate targets for.
      datasets:
        an iterable specifying the datasets on which the models are based. This
        controls the shape of the input images. Also indicates which weight file
        to use when loading weights from an external source.
      backends:
        an iterable of targets backends to generate targets for.
      reference_backend:
        the backend to use as a source of truth for the expected output results.
      failing_configurations:
        an iterable of dictionaries with the keys `models`, `datasets` and
        `backends`. Each key points to a string or iterable of strings
        specifying a set of models, datasets and backends that are failing.
      external_weights:
        a base url to fetch trained model weights from.
      tags:
        tags to apply to the test. Note that as in standard test suites, manual
        is treated specially and will also apply to the test suite itself.
      deps:
        test dependencies.
      size:
        size of the tests. Default: "large".
      python_version:
        the python version to run the tests with. Uses python3 by default.
      **kwargs:
        any additional arguments that will be passed to the underlying tests and
        test_suite.
    """
    failing_set = sets.make([])
    if failing_configurations != None:
        # Parse failing configurations.
        for configuration in failing_configurations:
            # Normalize configuration input.
            # {backend: "iree_llvmjit"} -> {backend: ["iree_llvmjit"]}
            for key, value in configuration.items():
                if type(value) == type(""):
                    configuration[key] = [value]

            for model in configuration["models"]:
                for dataset in configuration["datasets"]:
                    for backend in configuration["backends"]:
                        sets.insert(failing_set, (model, dataset, backend))

    tests = []
    for model in models:
        for dataset in datasets:
            for backend in backends:
                # Check if this is a failing configuration.
                failing = sets.contains(failing_set, (model, dataset, backend))

                # Append "_failing" to name if this is a failing configuration.
                test_name = name if not failing else name + "_failing"
                test_name = "{}_{}_{}__{}__{}".format(
                    test_name,
                    model,
                    dataset,
                    reference_backend,
                    backend,
                )
                tests.append(test_name)

                args = [
                    "--model={}".format(model),
                    "--data={}".format(dataset),
                    "--include_top=1",
                    "--reference_backend={}".format(reference_backend),
                    "--target_backends={}".format(backend),
                ]
                if external_weights:
                    args.append("--url={}".format(external_weights))

                # TODO(GH-2175): Simplify this after backend names are
                # standardized.
                # "iree_<driver>" --> "<driver>"
                driver = backend.replace("iree_", "")
                if driver == "llvmjit":
                    driver = "llvm"
                py_test_tags = ["driver={}".format(driver)]
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
                    main = "vision_model_test.py",
                    srcs = ["vision_model_test.py"],
                    args = args,
                    tags = py_test_tags,
                    deps = deps,
                    size = size,
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

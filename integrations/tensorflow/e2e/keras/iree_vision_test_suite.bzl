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

def iree_vision_test_suite(
        name,
        reference_backend,
        datasets,
        models_to_backends,
        external_weights = None,
        tags = None,
        deps = None,
        size = "large",
        python_version = "PY3",
        **kwargs):
    """Expands a set of iree_py_tests for the specified vision models and bundles them into a test suite.

    Creates one test per dataset, backend, and model.

    Args:
      name:
        name of the generated test suite.
      reference_backend:
        the backend to use as a source of truth for the expected output results.
      datasets:
        a list specifying the dataset on which the model is based. This controls
        the shape of the input images. Also indicates which weight file to use
        when loading weights from an external source.
      models_to_backends:
        a dictionary of models to lists of backends to run them on. Keys can
        either be tuples of strings (mapping multiple models to the same set of
        backends) or strings (mapping a single model to a set of backends).
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
    tests = []
    for models, backends in models_to_backends.items():
        if type(models) == type(""):  # Recommended style due to unstable API.
            models = (models,)
        for model in models:
            for backend in backends:
                for dataset in datasets:
                    test_backends = [reference_backend, backend]
                    test_name = "{}_{}_{}__{}".format(
                        name,
                        model,
                        dataset,
                        "__".join(test_backends),
                    )
                    tests.append(test_name)

                    args = [
                        "--model={}".format(model),
                        "--data={}".format(dataset),
                        "--include_top=1",
                        "--override_backends={}".format(",".join(test_backends)),
                    ]
                    if external_weights:
                        args.append("--url={}".format(external_weights))

                    iree_py_test(
                        name = test_name,
                        main = "vision_model_test.py",
                        srcs = ["vision_model_test.py"],
                        args = args,
                        tags = tags,
                        deps = deps,
                        size = size,
                        python_version = python_version,
                        **kwargs
                    )

    native.test_suite(
        name = name,
        tests = tests,
        # Note that only the manual tag really has any effect here. Others are
        # used for test suite filtering, but all tests are passed the same tags.
        tags = tags,
        # If there are kwargs that need to be passed here which only apply to
        # the generated tests and not to test_suite, they should be extracted
        # into separate named arguments.
        **kwargs
    )

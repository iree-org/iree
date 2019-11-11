# Copyright 2019 Google LLC
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

# TODO(laurenzo): This is available upstream as of 0.28. Remove when ready.
# See: https://docs.bazel.build/versions/master/repo/utils.html#maybe
def maybe(repo_rule, name, **kwargs):
    """Utility function for only adding a repository if it's not already present.
    This is to implement safe repositories.bzl macro documented in
    https://docs.bazel.build/versions/master/skylark/deploying.html#dependencies.
    Args:
        repo_rule: repository rule function.
        name: name of the repository to create.
        **kwargs: remaining arguments that are passed to the repo_rule function.
    Returns:
        Nothing, defines the repository when needed as a side-effect.
    """
    if not native.existing_rule(name):
        repo_rule(name = name, **kwargs)

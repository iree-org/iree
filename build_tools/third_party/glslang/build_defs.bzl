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

"""Build rules for utilizing glslang."""

def _glslang(name, mode = None, target = None, **kwargs):
    MODES = {
        "glsl": "",
        "hlsl": "-D",
    }
    if mode not in MODES:
        fail("Illegal mode {}".format(mode), "mode")

    TARGETS = {
        "opengl": "-G",
        "vulkan": "-V",
    }
    if target not in TARGETS:
        fail("Illegal target {}".format(target), "target")

    native.genrule(
        name = name,
        outs = [name + ".spv"],
        tools = ["@glslang//:glslangValidator"],
        cmd = ("$(location @glslang//:glslangValidator) " +
               MODES[mode] + " " + TARGETS[target] + ' "$(SRCS)" -o "$@"'),
        **kwargs
    )

def glsl_vulkan(name, **kwargs):
    _glslang(name, "glsl", "vulkan", **kwargs)

def hlsl_vulkan(name, **kwargs):
    _glslang(name, "hlsl", "vulkan", **kwargs)

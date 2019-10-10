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

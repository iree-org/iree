# IREE Repository Management

Due to the process by which we synchronize this GitHub project with our internal
Google source code repository, there are some oddities in our workflows and
processes. We aim to minimize these, and especially to mitigate their impact on
external contributors, but they are documented here for clarity and
transparency. If any of these things are particularly troublesome or painful for
your workflow, please reach out to us so we can prioritize a fix.

Hopefully these quirks actually make usage in other downstream projects easier,
but integrators may need to look past some details (like the Bazel build system,
Android support, etc.) based on their specific requirements.

## Build Systems

IREE supports building from source with both Bazel and CMake. CMake is the
preferred build system for open source users and offers the most flexible
configuration options. Bazel is a stricter build system and helps with usage in
the Google internal source repository. Certain dependencies (think large/complex
projects like CUDA, TensorFlow, PyTorch, etc.) may be difficult to support with
one build system or the other, so the project may configure these as optional.

# CMake Presets

We are experimenting with CMake presets for setting up the project in
various ways. Because these can interact badly with tools, we do not
yet have a `CMakePresets.json` in the root project folder, instead
requiring someone to add one themselves and include presets from here.

Once this is tested some more, we will enable by default. For now,
create a `CMakeUserPresets.json` in the project root and put this in it:

```
{
    "version": 4,
    "cmakeMinimumRequired": {
      "major": 3,
      "minor": 23,
      "patch": 0
    },
    "include": [
        "build_tools/cmake/presets/all.json"
    ]
}
```

## Using Presets

There are two variants of presets that we use:

* `new-{os}-{variant}`: Sets up a new dev directory as documented
[on the website](https://iree.dev/building-from-source/getting-started/#quickstart-clone-and-build)
with various further options. This is run from a source directory and will
set up the generator and build into `../iree-build`.
* `{variant}`: The raw variant options can be set at any time from
a build directory with `cmake --preset {variant} .`.

Since the 'new' variants are OS specific, you must use the one appropriate
for your operating system. One of "linux", "macos", "windows".

## New Project Presets

* `cmake --preset new-{os}-dev`: Start a new dev build with CMake defaults for
  all project features.
* `cmake --preset new-{os}-minimal`: Start a new dev build with all optional 
  project features disabled. While not particularly useful, this lets selected
  features be enabled as needed.
* `cmake --preset new-{os}-turbine`: Start a new dev build with all optional
  project features disabled except those needed to do Turbine development (
  currently CPU backend, Torch input, and Python bindings).

## Existing Project Presets

An existing build directory can have its CMake options redefined with
these presets:

* `cmake . --preset minimal`: Disable all optional project features.
* `cmake . --preset python-bindings`: Enable Python bindings in dev mode.
* `cmake . --preset turbine`: Disable all optional project features except
  those needed for Turbine development on CPU.

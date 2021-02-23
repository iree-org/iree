---
layout: default
permalink: debugging/releases
title: "Debugging releases playbook"
---

# Debugging releases playbook
{: .no_toc }

## Tools and Locations

* `.github/workflows/build_package.yml`: Release packaging jobs
* `build_tools/github_actions/build_dist.py`: Main script to build various
  release packages (for all platforms). We usually use this when reproing to
  approximate exactly what the CI does. Assumes a subdirectory of `main_checkout`
  and writes builds to `iree-build` and `iree-install` as a peer of it. To use
  locally, just symlink your source dir as `main_checkout` in an empty
  directory (versus checking out).

## Manylinux releases

The Linux releases are done in a manylinux2014 docker container. At the time of
this writing, it has gcc 9.3.1 and Python versions 3.5 - 3.9 under `/opt/python`.
Note that this docker image approximates a 2014 era RHEL distro, patched with
backported (newer) dev packages. It builds with gcc and BFD linker unless if
you arrange otherwise. `yum` can be used to get some packages.

Get a docker shell (see exact docker image in build_package.yml workflow):

```shell
docker run --rm -it -v $(pwd):/work/main_checkout stellaraccident/manylinux2014_x86_64-bazel-3.7.2:latest /bin/bash
```

Remember that docker runs as root unless if you take steps otherwise. Don't
touch write files in the `/work/main_checkout` directory to avoid scattering
root owned files on your workstation.

The default system Python is 2.x, so you must select one of the more modern
ones:

```shell
export PATH=/opt/python/cp39-cp39/bin:$PATH
```


Build core installation:

```shell
# (from within docker)
cd /work
python ./main_checkout/build_tools/github_actions/build_dist.py main-dist

# Also supports:
#   main-dist
#   py-runtime-pkg
#   py-xla-compiler-tools-pkg
#   py-tflite-compiler-tools-pkg
#   py-tf-compiler-tools-pkg
```

You can `git bisect` on the host and keep running the above in the docker
container. Note that every time you run `build_dist.py`, it deletes the cmake
cache but otherwise leaves the build directory (so it pays the configure cost
but is otherwise incremental). You can just `cd iree-build` and run `ninja`
for faster iteration (after the first build or if changing cmake flags).
Example:

Extended debugging in the manylinux container:

```shell
cd /work/iree-build
# If doing extended debugging in the container, these may make you happier.
yum install ccache devtoolset-9-libasan-devel gdb

# Get an LLVM symbolizer.
yum install llvm9.0
ln -s /usr/bin/llvm-symbolizer-9.0 /usr/bin/llvm-symbolizer

# You can manipulate cmake flags. These may get you a better debug experience.
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASAN=ON -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=gold -DIREE_ENABLE_CCACHE=ON .

ninja

# Or you may need this if buggy LLVM tools (like mlir-tblgen) are leaking :(
ASAN_OPTIONS="detect_leaks=0" ninja
```

Other tips:

* If debugging the runtime, you may have a better time just building the
  Release mode `main-dist` package above once, which will drop binaries in the
  `iree-install` directory. Then build the `py-runtime-pkg` or equiv and
  iterate further in the build directory. Ditto for TF/XLA/etc.
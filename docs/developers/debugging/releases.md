# Debugging releases playbook

## Tools and Locations

* `.github/workflows/build_package.yml`: Release packaging jobs
* `build_tools/github_actions/build_dist.py`: Main script to build various
  release packages (for all platforms). We usually use this when reproing to
  approximate exactly what the CI does. Assumes a subdirectory of `c`
  and writes builds to `iree-build` and `iree-install` as a peer of it. To use
  locally, just symlink your source dir as `c` in an empty
  directory (versus checking out).

## Mapping releases back to git commits

The source IREE commit SHA is embeded into pip releases in a few places.
Starting in a python venv, you can find the IREE commit from both the shell:

```shell
"$(find . -name 'iree-compile' -executable)" --version
IREE (https://openxla.github.io/iree):
  IREE compiler version 20231016.553 @ f1cb2692a086738d7f16274b9b3af6d2c15ef133
  LLVM version 18.0.0git
  Optimized build
```

and the Python API:

```shell
python -c "import iree.compiler.version as v; print(v.REVISIONS['IREE'])"
f1cb2692a086738d7f16274b9b3af6d2c15ef133
``` 

## Manylinux releases

The Linux releases are done in a manylinux2014 docker container. At the time of
this writing, it has gcc 9.3.1 and Python versions 3.5 - 3.9 under `/opt/python`.
Note that this docker image approximates a 2014 era RHEL distro, patched with
backported (newer) dev packages. It builds with gcc and BFD linker unless if
you arrange otherwise. `yum` can be used to get some packages.

Get a docker shell (see exact docker image in build_package.yml workflow):

```shell
docker run --rm -it -v $(pwd):/work/c stellaraccident/manylinux2014_x86_64-bazel-4.2.2:latest /bin/bash
```

Remember that docker runs as root unless if you take steps otherwise. Don't
touch write files in the `/work/c` directory to avoid scattering
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
python ./c/build_tools/github_actions/build_dist.py main-dist

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
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_ENABLE_ASAN=ON -DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=gold -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache .

ninja

# Or you may need this if buggy LLVM tools (like mlir-tblgen) are leaking :(
ASAN_OPTIONS="detect_leaks=0" ninja
```

Other tips:

* If debugging the runtime, you may have a better time just building the
  Release mode `main-dist` package above once, which will drop binaries in the
  `iree-install` directory. Then build the `py-runtime-pkg` or equiv and
  iterate further in the build directory. Ditto for TF/XLA/etc.

## Testing releases on your fork

To avoid interrupting the regular releases published on the IREE github, you
can test any changes to the release process on your own fork.  Some setup is
required before these github actions will work on your fork and development
branch.

You can run
[`schedule_candidate_release.yml`](https://github.com/openxla/iree/blob/main/.github/workflows/schedule_candidate_release.yml)
with a workflow dispatch from the actions tab. If you want to test using a
commit other than the latest green on your `main` branch, modify the section
that
[identifies the latest green commit](https://github.com/openxla/iree/blob/c7b29123f8bd80c1346d2a9e6c5227b372b75616/.github/workflows/schedule_candidate_release.yml#L25)
to search from another commit or just hardcode one.

To speed up
[`build_package.yml`](https://github.com/openxla/iree/blob/main/.github/workflows/build_package.yml),
you may want to comment out some of the builds
[here](https://github.com/openxla/iree/blob/392449e986493bf710e3da637ebf807715da9ffe/.github/workflows/build_package.yml#L34-L87).
The
[`py-pure-pkgs`](https://github.com/openxla/iree/blob/392449e986493bf710e3da637ebf807715da9ffe/.github/workflows/build_package.yml#L52)
build takes only ~2 minutes and the
[`py-runtime-pkg`](https://github.com/openxla/iree/blob/392449e986493bf710e3da637ebf807715da9ffe/.github/workflows/build_package.yml#L39)
build takes ~5, while the others can take several hours.

From your development branch, you can manually run the
[Schedule Snapshot Release](https://github.com/openxla/iree/actions/workflows/schedule_snapshot_release.yml)
action, which invokes the
[Build Release Packages](https://github.com/openxla/iree/actions/workflows/build_package.yml)
action, which finally invokes the
[Validate and Publish Release](https://github.com/openxla/iree/actions/workflows/validate_and_publish_release.yml)
action.  If you already have a draft release and know the release id, package
version, and run ID from a previous Build Release Packages run, you can
also manually run just the Validate and Publish Release action.

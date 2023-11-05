# CMake with `ccache`

[`ccache`](https://ccache.dev/) is a compilation cache. In principle, just
prepending compiler invocations with `ccache` is all one needs to enable it,
e.g.

```shell
ccache clang foo.c -c -o foo.o
```

takes care of executing `clang` with these arguments and caches the output file
`foo.o`. The next invocation then skips executing `clang` altogether.

When the cache is hit, the speedup is such that the "compilation" becomes
essentially free. However, `ccache` only caches compilation,
[not linking](https://stackoverflow.com/a/29828811).

Here a few scenarios where `ccache` helps:

* Incremental rebuilds. While `cmake` always tries to avoid unnecessary work in
  incremental rebuilds, it can only make simple decisions based on file
  timestamps. `ccache` sees deeper: if the raw source code isn't readily
  a cache hit, it will then try again after preprocessing and discarding
  comments.
* One pain point with `cmake` is having to start over from a clean build
  directory from time to time, which by default means paying again the full cost
  of a cold build. Thankfully `ccache` keeps its cache outside of any `cmake`
  build directory, so the first build in the new clean build directory may be
  very fast.

## Installing and setting up `ccache`

`ccache` is available on most platforms. On Debian-based Linux distributions,
do:

```shell
sudo apt install ccache
```

The one `ccache` setting that you probably need to configure is the maximum
cache size. The default `5G` is too small for our purposes. To set the cache max
size, do this once:

```shell
ccache --max-size=20G
```

**Tip:** At the moment (late 2020), most of the code we're building is
`third_party/llvm-project` so the fundamental limiting factor to how far we can
cache away rebuilds is how often that dependency gets updated. Given how
frequently it currently is updated, I'm finding that `20G` is enough to make the
`ccache` size not be the limiting factor.

## Telling CMake to use `ccache`

Use the CMake
[COMPILER_LAUNCHER functionality](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER_LAUNCHER.html)
by setting `CMAKE_C_COMPILER_LAUNCHER=ccache` and
`CMAKE_CXX_COMPILER_LAUNCHER=ccache` in your

Notes:

* This approach only works with the `Ninja` and `Makefile` generators
  (`cmake -G` flag). When using other generators, another approach is needed,
  based on wrapping the compiler in a script that prepends `ccache`. See this
  [article](https://crascit.com/2016/04/09/using-ccache-with-cmake/).

## Ensuring that `ccache` is used and monitoring cache hits

The `ccache -s` command dumps statistics, including a cache hit count and ratio.
It's convenient to run periodically with `watch` in a separate terminal:

```shell
watch -n 0.1 ccache -s  # update the stats readout every 0.1 seconds
```

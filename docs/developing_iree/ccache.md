# Using `ccache` to build IREE

[`ccache`](https://ccache.dev/) is a compilation cache. In principle, just prepending compiler invocations with `ccache` is all one needs to enable it, e.g.
```shell
ccache clang foo.c -c -o foo.o
```
takes care of executing `clang` with these arguments and caches the output file `foo.o`. The next invocation then skips executing `clang` altogether.

## Installing and setting up `ccache`

`ccache` should be easy to install about anywhere. On Debian-based Linux distributions, do:
```shell
sudo apt install ccache
```

The one `ccache` setting that you probably need to configure is the maximum cache size. The default `5G` is too small for our purposes. To set the cache max size, do this once:
```shell
ccache --max-size=20G
```

**Tip:** At the moment (late 2020), most of the code we're building is `third_party/llvm-project` so the fundamental limiting factor to how far we can cache away rebuilds is how often that dependency gets updated. Given how frequently it current is update, I'm finding that `20G` is enough to make the `ccache` size not be the limiting factor.

## Telling CMake to use `ccache`

This is actually a complex topic, there are two different approaches here as explained in this [article](https://crascit.com/2016/04/09/using-ccache-with-cmake/).

### Suggested approach: pass CMake options

In the initial CMake configuration step, set the `IREE_ENABLE_CCACHE` and `LLVM_CCACHE_BUILD` options, like this:
```shell
cmake -G Ninja \
  -DIREE_ENABLE_CCACHE=ON \
  -DLLVM_CCACHE_BUILD=ON \
  ... other options as usual
```

Notes:
* In the above command line, I explicitly put the `-G Ninja` flag even though that's not a new flag, we're always using `Ninja`. The reason is that the whole approach that these options are relying on only works on certain generators --- `Ninja` and `Makefile`. So if we ever used different generators, we'd have to switch to the other approach, as explained in the above linked [article](https://crascit.com/2016/04/09/using-ccache-with-cmake/).
* We do need two separate options `IREE_ENABLE_CCACHE` and `LLVM_CCACHE_BUILD`, because of how CMake leaves it up to each project to define how to control the use of `ccache`, and `llvm-project` is a `third_party/` project in IREE. Note that most of the compilation time is spent in `llvm-project`, so `LLVM_CCACHE_BUILD` is the most important flag here.

### Other approach: wrapper scripts

This is how `ccache` was traditonally used with `make`. This approach is well explained in the above linked [article](https://crascit.com/2016/04/09/using-ccache-with-cmake/). The main reason why you might need it is if you were using another CMake generator than `Ninja` or `Makefile`. For instance, that article covers the case of using the `XCode` generator.

This approach comes with its own caveats:
* We have both `C` and `C++` code, so you need at least two wrapper scripts, for the `C` and `C++` compilers.
* For cross-compilation builds, e.g. Android builds, where there is a "device toolchain" separate from the "host toolchain", you will need wrapper scripts for both toolchains, bringing the number of required scripts to 4.
* As explained in the above linked [article](https://crascit.com/2016/04/09/using-ccache-with-cmake/), pointing CMake to the right compiler is itself a little generator-dependent, e.g. `CMAKE_XCODE_ATTRIBUTE_CC` when using the `XCode` generator.

## Ensuring that `ccache` is used and monitoring cache hits

The `ccache -s` command dumps statistics, including a cache hit count and ratio. It's convenient to run periodically with `watch` in a separate terminal:
```shell
watch ccache -s  # defaults to running it every 2 seconds
```

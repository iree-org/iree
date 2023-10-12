# CPU cache and other CPU event profiling

CPUs are able to
[record](https://en.wikipedia.org/wiki/Hardware_performance_counter) certain
events that may be relevant when investigating the performance of a program. A
common example of such an event is a ["cache
miss"](https://en.wikipedia.org/wiki/CPU_cache#Cache_miss), when the program
tries to access data in memory that isn't already in some CPU cache, causing
that access to be slower than it could otherwise be.

Querying and analyzing this data can be useful, but is hard in two distinct
ways:

* Depending on the CPU and on the OS, both hardware and software limitations can
  get in the way of obtaining accurate data.
* This data tends to be inherently difficult to interpret, even when it is
  perfectly accurate. In practice it is often noisy and inaccurate, which makes
  interpretation even more complicated.

There are two parts to this page: platform-specific information about [how to
query](#perf-and-simpleperf-on-linux-and-android) this data, and, at the end, a
platform-independent explanation of [how to
interpret](#interpreting-cpu-event-counts) it.

## Perf and Simpleperf, on Linux and Android

### Overview

The Linux kernel exposes system event counters to user-space programs by means
of the
[`perf_event_open`](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)
system call. This includes both hardware event counters (such as CPU cache
events) and software events from the kernel (such as page faults and context
switches). Anyone may use this system call to implement a profiler, but Linux
readily offers one, [`perf`](https://perf.wiki.kernel.org/index.php/Main_Page).

### Preserving Artifacts

By default IREE cleans up any temporary files it creates while running. Tools
like perf, however, require those files exist even after the process has exited.
The environment variable `IREE_PRESERVE_DYLIB_TEMP_FILES` can be set to preserve
the files. This is only needed for the CPU path when using the system loader.

```shell
export IREE_PRESERVE_DYLIB_TEMP_FILES=1
```

### Desktop Linux

On desktop Linux we can use
[`perf`](https://perf.wiki.kernel.org/index.php/Main_Page). It is provided on
most Linux distributions, for instance on Debian-based distributions do:

```shell
sudo apt install linux-perf
```

Run the program to be profiled, prepending its command line with `perf record`.
By default this will write the profile data to the current directory,
`./perf.data`. Sometimes this isn't ideal, such as then the current directory is
under version control. Explicit paths can be specified by `-o` flag to direct
the output of `perf record`, and then by `-i` flags to select the input of
subsequent commands analyzing the profile. Example:

```shell
perf record -o /tmp/perf.data \
  ./tools/iree-benchmark-module \
    --device=local-task \
    ... command-line arguments of iree-benchmark-module as usual ...
```

By default, this samples time spent. One may specify instead an event to sample
by, with the `-e` flag. For instance, to sample by L1 cache misses, one may do:

```shell
perf record -o /tmp/perf.data -e L1-dcache-load-misses \
  ./tools/iree-benchmark-module \
    --device=local-task \
    ... command-line arguments of iree-benchmark-module as usual ...
```

`perf list` dumps the list of event types.

Once you have recorded a profile, there are two main ways to analyze it: `perf
report` and `perf annotate`.

`perf report` breaks down the event counts by symbol. In the default case where
what was sampled was time, this is just an ordinary profile by symbol name, no
different than what could be viewed in other profilers such as
[Tracy](profiling_with_tracy.md). Where it gets really interesting is when the
profile was recording a specific event type, as in the above `-e
L1-dcache-load-misses` example:

``` shell
perf report -i /tmp/perf.data

Samples: 6K of event 'L1-dcache-load-misses', Event count (approx.): 362571861
Overhead  Command          Shared Object              Symbol
  61.53%  cpu0             dylib_executablenzpx2Q.so  [.] serving_default_ex_dispatch_31
  13.30%  cpu0             dylib_executablenzpx2Q.so  [.] serving_default_ex_dispatch_11
   2.11%  cpu0             dylib_executablenzpx2Q.so  [.] serving_default_ex_dispatch_13
   1.90%  cpu0             dylib_executablenzpx2Q.so  [.] serving_default_ex_dispatch_19
   1.54%  cpu0             dylib_executablenzpx2Q.so  [.] serving_default_ex_dispatch_25
   1.49%  cpu0             dylib_executablenzpx2Q.so  [.] serving_default_ex_dispatch_5
```

`perf annotate` breaks down the event counts by instruction. Again, in the
default case where what was sampled was time, this is no different than what
could be viewed in Tracy, and the real motivation to use `perf` is when
profiling by specific event types as in the above `-e L1-dcache-load-misses`
example:

``` shell
perf annotate -i perf.data

Samples: 6K of event 'L1-dcache-load-misses', 4000 Hz, Event count (approx.): 362571861
serving_default_ex_dispatch_31  /tmp/dylib_executablenzpx2Q.so [Percent: local period]
  1.66 │        movups -0x1000(%rdi),%xmm10
  0.48 │        movups -0x800(%rdi),%xmm9
  0.82 │        movups (%rdi),%xmm8
  0.49 │        movaps %xmm1,%xmm4
  0.12 │        shufps $0x0,%xmm1,%xmm4
  0.14 │        mulps  %xmm5,%xmm4
  0.28 │        addps  %xmm6,%xmm4
  0.60 │        movaps %xmm3,%xmm6
  0.34 │        shufps $0x0,%xmm3,%xmm6
```

#### Warning

`perf annotate` is even noisier than `perf report` as it can be overly
optimistic, depending on the CPU, to pin an event to a specific instruction.
Typically, this works fairly well on x86 CPUs and less well on ARM CPUs and more
generally on anything mobile. Even on a desktop x86 CPU, this is noisy, as the
above example (recorded on a Skylake workstation) shows: it blamed a `mulps
%xmm5,%xmm4` instruction for a cache miss, which doesn't make sense as that
instruction only touches registers.

### Android

On Android we can use
[`simpleperf`](https://developer.android.com/ndk/guides/simpleperf). It's
preinstalled on current Android `userdebug` images, and part of the Android NDK.

In theory, as Android is Linux, it should be possible to use `perf`.
Unfortunately, `perf` is difficult to build for Android. Fortunately,
`simpleperf` is readily available: it is preinstalled in Android `userdebug`
images, and it is part of the Android NDK.

First, we record on the device:

```shell
adb shell \
  simpleperf record -e raw-l1d-cache-refill -o /data/local/tmp/perf.data \
    /data/local/tmp/iree-benchmark-module \
      --device=local-task \
      ... command-line arguments of iree-benchmark-module as usual ...
```

Then pull the recorded data from the device, and analyze on the desktop. We
assume that `${ANDROID_NDK}` points to the local copy of the Android NDK.

```shell
adb pull /data/local/tmp/perf.data /tmp/perf.data
${ANDROID_NDK}/simpleperf/report.py -i /tmp/perf.data
```

This prints a breakdown of `raw-l1d-cache-refill` events by symbol.

Like with `perf`, a list of event types can be queried by the `list` subcommand:

```shell
adb shell simpleperf list
```

#### No support for `annotate` by CPU event

There is no `simpleperf annotate`. The `simpleperf` documentation lists a couple
of
[ways](https://android.googlesource.com/platform/system/extras/+/master/simpleperf/doc/README.md#show-annotated-source-code-and-disassembly)
of achieving the same thing.

However:

* The common case of annotating by time, as opposed to annotating by CPU event,
  is supported by [Tracy](profiling_with_tracy.md).
* Annotating by CPU event is inherently not working due to hardware limitations
  of the ARM CPUs found in Android devices. That is, the hardware is too
  imprecise at pinning an event to a particular instruction.

## Interpreting CPU event counts

### Problems

There are multiple layers of complexity in interpreting CPU event counts.

#### These events are in themselves normal

The first difficulty is in the fact that most of these events are *normal*. So
just knowing that they happened is not in itself actionable.

For example, if we learn that some code causes cache misses, that isn't big
news: so does all code. Maybe this code has *too many* cache misses, but how
many is too many? Maybe this code alone accounts for a large fraction of the
overall total of the whole program, but maybe even that is normal, for instance
if the code being studied is the 'hot' part of the program where a large
fraction of overall time is spent?

#### These events are hardware-dependent and under-documented

Many of these events have a meaning that varies between CPUs and that is
difficult to characterize on any CPU, let alone in a way that applies to all
CPUs.

For example, take the "L2 data cache refill". On ARM, with `simpleperf`, that
would be `raw-l2d-cache-refill`. Questions:

* Is “L2” [inclusive](https://en.wikipedia.org/wiki/Cache_inclusion_policy) of
  “L1”?
* How many bytes are transferred per “refill”?
* Are accesses induced by speculative execution or by automatic pre-fetching
  counted in the same way as accesses induced by actual code execution?

The answers to all of the above questions are CPU-dependent. They may even vary
between the CPU cores of the same Android device.

#### These events are imprecise and noisy, particularly on ARM CPUs

Expect noise levels above 10% in many CPU event counts on ARM CPUs. Moreover, on
ARM, as discussed above, there is inaccuracy in which instruction is blamed for
which event, which will increase inaccuracy of per-symbol breakdowns for very
cheap symbols (and makes `perf annotate` impossible as noted above). Finally, be
aware that some ARM CPUs may perform event count interpolation, so we may not
have any access to true hardware counts.

### Recommendations

Here is a workflow pattern that allows to make significant use of CPU event
counts, despite all the problems noted above:

* Hypothesize that some code diff might help performance, and might help
  reducing the number of CPU events of a certain type, and that the two might be
  related.
* Benchmark with and without the code diff, on the same device, everything else
  being equal.
    * Let your benchmark perform a fixed number of iterations, or, if using a
    benchmark termination condition of the form "run until at least N seconds
    have elapsed", carefully divide event counts by the actual number of
    iterations that were run.
* If the observed CPU event count difference is significant, go ahead and claim
  that your code diff probably helps with that aspect of CPU behavior.

Some things NOT to be done:

* Don’t try to compare different metrics, not even when it seems obvious that
  they should satisfy a simple relationship, not even on the same CPU (e.g. “L1
  accesses should be greater than L2 accesses”).
* Don’t divide by some “total” metric to get some kinds of ratios. For example,
  don’t try to compute a “cache miss ratio” as quotient of “cache refill” over
  “all cache accesses” metrics. The first problem with that (even before we get
  to CPU-specific issues) is that that’s rewarding increases to the “all cache
  accesses” metrics, so if something bad happens in your codegen and your kernel
  ends up spilling a lot of register to the stack, that’s going to be a lot more
  accesses which will all be L1 hits so that’ll help this ratio look better!  So
  more generally, just try to minimize some CPU metrics (that count “costly”
  events), not some more complex math expression formed from arithmetic on CPU
  metrics.

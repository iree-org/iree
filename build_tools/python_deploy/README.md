# Python Deployment

See comments in scripts for canonical usage. This page includes additional
notes.

## Debugging manylinux builds

We build releases under a manylinux derived docker image. When all goes well,
things are great, but when they fail, it often implicates something that has
to do with Linux age-based arcana. In this situation, just getting to the
shell and building/poking can be the most useful way to triage.

Here is the procedure:

```
[host ~/iree]$ docker run --interactive --tty --rm -v $(pwd):/work/iree gcr.io/iree-oss/manylinux2014_x86_64-release:prod
[root@c8f6d0041d79 /]# export PATH=/opt/python/cp310-cp310/bin:$PATH
[root@c8f6d0041d79 /]# python --version
Python 3.10.4


# Two paths for further work.
# Option A: Build like a normal dev setup (i.e. if allergic to Python
# packaging and to triage issues that do not implicate that).
[root@c8f6d0041d79 ]# cd /work/iree
[root@c8f6d0041d79 iree]# pip install wheel cmake ninja pybind11 numpy
[root@c8f6d0041d79 iree]# cmake -GNinja -B ../iree-build/ -S . -DCMAKE_BUILD_TYPE=Release -DIREE_BUILD_PYTHON_BINDINGS=ON
[root@c8f6d0041d79 iree]# cd ../iree-build/
[root@c8f6d0041d79 iree-build]# ninja

# Options B: Creates Python packages exactly as the CI/scripts do.
# (to be used when there is Python arcana involved). The result is organized
# differently from a usual dev flow and may be subsequently more confusing to
# the uninitiated.
[root@c8f6d0041d79 iree]# pip wheel compiler/
[root@c8f6d0041d79 iree]# pip wheel runtime/
```

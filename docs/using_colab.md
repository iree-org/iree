# Using Colab

Since so many ML frameworks are Python based, we often use Colab for interactive
programming. With IREE's Python bindings, it becomes possible to quickly proof
out e2e flows that cross the ML framework, compiler and runtime interactively.

## Quick Start

Run:

```shell
./colab/start_colab_kernel.py
```

This will start a jupyter notebook on port 8888. Then navigate to
[Google's Colab Site](https://colab.research.google.com), create or open a
Python3 notebook and Connect to a Local Runtime.

There are some sample notebooks in the colab/ directory (which you can either
"upload" or load from GitHub in the File menu).

Note that sometimes, if you opt in to new Colab features, your URL bar will
actually be for "colab.sandbox.google.com" instead of
"colab.research.google.com". If so, you will be unable to connect to the local
runtime, which is hard-coded to allow the "colab.research.google.com" domain.
There will typically be a box at the top in such situations which lets you opt
to use the production release.

## Installation

Unless if you have setup Colab before, there is some setup:

### Install Jupyter (from https://jupyter.org/install)

```shell
python3 -m pip install --upgrade pip
python3 -m pip install jupyter
```

### Setup colab (https://research.google.com/colaboratory/local-runtimes.html)

```shell
python3 -m pip install jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
```

## Local and Hosted Runtimes

IREE's Python bindings are not yet published in an accessible way, so they must
be bundled within the runtime binary itself. Due to this, you must use a local
runtime started as described above in order to execute IREE's Colab notebooks.
Hosted/remote runtimes are not yet supported, though we have plans to improve
this process as the project matures.

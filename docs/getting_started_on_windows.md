# Getting Started on Windows

There are multiple ways to develop on Windows, and compared to other
environments, batteries are often not included. This document outlines the setup
that is known to work for the IREE developers. Other setups almost certainly
work as well.

## Pre-requisites

### Enable Developer Mode

We use symlinks in the build, and this requires developer mode to be enabled:

*   Open Settings
*   Search for "Developer Settings"
*   Select the radio button "Developer mode" and accept all prompts

### Install Build Tools For Visual Studio

At the time of writing, this can be found by going to
[this page](https://visualstudio.microsoft.com/downloads/) and finding the
Download link for "Build Tools for Visual Studio 2019".

Minimally select to install "C++ Build Tools".

### Install the Scoop Package Manager

The [Scoop page is here](https://scoop.sh/). Follow the instructions at the
bottom.

IMPORTANT: When launching PowerShell, make sure to not select any option that
ends in "(x86)" as this will install the 32bit version, and all of the software
that Scoop installs will be 32bit. While 32bit builds of the project may be
possible, we only (presently) support 64bit. Also, not all packages are
available in 32bit.

## Install Scoop Packages

### Home Directory

Optional: Set the msys2 home directory to your Windows home directory. By
default, the msys2 HOME will be nested inside the installation directory, which
is managed by scoop and somewhat ephemeral.

*   Open the Windows environment dialog (This PC -> Properties -> Advanced
    Settings -> Environment Variables...)
*   Add a new User variable: `HOME` = `C:\Users\%USERNAME%` (replacing with the
    actual location of your Windows home directory).
*   Ok out of all dialogs
*   Restart PowerShell
*   Verify that it took effect (`Get-ChildItem Env:HOME`)

While here, also consider adding `GIT_SSH=C:/Users/%USERNAME%/scoop/shims/ssh`
which will help git find the right SSH and keys (you probably want this as a
global to Windows vs just in your shell, which is why it is recommended here).

### Scoop Package Setup

In PowerShell, run the following:

```shell
scoop install git nano vim
scoop bucket add extras
scoop bucket add versions
scoop install msys2
scoop install curl cmake openssh python36 llvm bazel
# Optional
scoop install vscode
```

TODO: Upgrade to head python.

Check the .bazelversion file for the version of Bazel you should install. You
can also use [Bazelisk](https://github.com/bazelbuild/bazelisk) to manage Bazel
versions.

TODO: Specific steps to install a compatible version of Bazel directly

## Setup MSYS2

Then run msys2 for the first time for subsequent setup by doing one of:

*   Launch MSYS2 from the start menu
*   From a regular PowerShell terminal (not ISE, which hangs), run `msys2`
*   Note that full interop between native and msys2 programs requires a "native"
    Windows shell. "PowerShell" qualifies. "PowerShell ISE" and the way that the
    "MSYS2" system shortcut launch do not. You can tell if things are working by
    launching MSYS2 and running `python` (which, for us is a native windows
    application).
*   Reportedly, cmder is a good possibility.

You can also customize you shell, etc. From now on, when we refer to the shell,
we mean "launch msys2".

```shell
# In MSYS2 shell
pacman -S patch
# Customize the path to your home directory if required.
echo 'export PATH=/c/Users/$USERNAME/scoop/shims:$PATH' >> ~/.bash_profile
```

You are also going to want a few other environment variables. You are welcome to
configure these however you choose. Adding them to `~/.bash_profile` would look
like this:

```shell
# Tells Bazel to use clang-cl instead of VS cl.
export USE_CLANG_CL=1
# Tells Bazel where the LLVM installation is.
export BAZEL_LLVM=C:/Users/$USERNAME/scoop/apps/llvm/current
# This should be automatic, but worth checking.
export BAZEL_SH=C:/Users/$USERNAME/scoop/apps/msys2/current/usr/bin/bash.exe
```

## Install the Vulkan SDK

Some parts of the project link against the Vulkan SDK and require it be
installed on your system. If you are planning on building these, or see linker
errors about undefined references to `vk` symbols, download and install the
Vulkan SDK from https://vulkan.lunarg.com/, and check that the VULKAN_SDK
environment variable is set when you are building.

## Optional: Configure Git

### Git SSH

*   Make sure that the environment variable is set:
    `GIT_SSH=C:/Users/%USERNAME%/scoop/shims/ssh`
*   Generate SSH Key: `ssh-keygen -t rsa -b 4096 -C "EMAIL@email.com"`
*   Add `~/.ssh.id_rsa.pub` key to GitHub
*   Try a test connection `ssh git@github.com`

### Other git config options

```shell
# Disable stupid^H^H^H^H^H^H Windows line ending translation
git config --global core.autocrlf true
# Configure name and email before commiting anything
git config --global user.name "MY NAME"
git config --global user.email "MY EMAIL"
```

## Clone and Build

This assumes that we are cloning into `C:\src\ireepub`. Update accordingly for
your use.

### Clone

Note that if you will be cloning frequently, it can be sped up significantly by
creating a reference repo and setting
`IREE_CLONE_ARGS="--reference=/path/to/reference/repo"`. See
`scripts/git/populate_reference_repo.sh` for further details.

```shell
IREE_CLONE_ARGS=""
mkdir -p /c/src/ireepub
cd /c/src/ireepub
git clone $IREE_CLONE_ARGS https://github.com/google/iree.git iree
cd iree
git submodule init
git submodule update $IREE_CLONE_ARGS --recursive
```

### Build

```shell
# Run all core tests
bazel test -k --config=windows iree/...
```

In general, build artifacts will be under the `bazel-bin` directory at the top
level.

## Recommended user.bazelrc

You can put a user.bazelrc at the root of the repository and it will be ignored
by git. The recommended contents for Windows are:

```
build --config=windows
build --disk_cache=c:/bazelcache
build:debug --compilation_mode=dbg --copt=/O2 --per_file_copt=iree@/Od --strip=never
```

## Troubleshooting

### Bazel and Visual Studio Versions

In general, the software has only been build with Visual Studio 2019 Build Tools
and Clang-CL 9.x. Previous versions are known to have incompatibilities in their
standard libraries. If you have multiple versions of Visual Studio (and/or Build
Tools) installed, Bazel may auto-detect the wrong one. You can see this by
adding a `-s` argument to you build command and looking for a "SET INCLUDE="
line in the log output (to see where it is pointing).

You can hard-code the version that Bazel selects by setting:
`BAZEL_VS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools`.

If updating, you will need to `bazel clean` and `bazel shutdown` for changes to
take effect.

See
[this page](https://docs.bazel.build/versions/master/windows.html#build-c-with-msvc)
for more options.

If setting up a new machine, it is best to just make sure there is one version.

### Caching

Bazel can use a local disk cache, which can speed up compiles that iterate
between different sets of flags (ie. optimized and prod). Add this to your
user.bazelrc:

```
build --disk_cache=c:/bazelcache
```

### Debugging

By default, the project builds in opt mode, which is optimized/stripped. For
Windows builds that do not wish to switch entirely to a debug build (i.e. it is
often advantageous to only disable optimizations for some part of the code you
are working on, you can use a build config like this by adding it to your
user.bazelrc and building with --config=debug):

```
build:debug --compilation_mode=dbg --copt=/O2 --per_file_copt=iree@/Od --strip=never
```

Note that there is a Windows specific sharp edge: The `-O0` flag does nothing on
CL-like compilers. You must use the Microsoft syntax of /Od. Given that, we just
use it consistently.

## Annex

### Configuring Python

The python bindings are still rudimentary but do require a functioning Python
install with deps. If you installed Python from scoop or another place that
doesn't bundle common deps, you'll need to take a couple of extra steps:

#### Install PIP:

Try running `pip`. If it doesn't exist (and if it isn't in your python Scripts/
directory and somehow excluded from your path), install it by:

```shell
which python
# Verify that this is where you think it is. Also verify that pip
# prints install paths where you think.
curl https://bootstrap.pypa.io/get-pip.py > ~/Downloads/get-pip.py
python ~/Downloads/get-pip.py
# Note that pip may print a directory name that needs to be added
# to the path. Do so.
```

#### Install Python Deps

```shell
pip install numpy
```

If using Colab, you may also want to install TensorFlow:

```shell
pip install tf-nightly
```

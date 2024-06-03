# Virtual machine (VM)

## Overview

The VM is an abstract machine that defines a type system of primitive and
reference objects, module machinery, and a fairly involved mechanic for calls
and dynamically binding extern funcs to their defs.

It comes with a bytecode module type, which implements the module interface and
exposes a CFG based instruction set (sans pointers, as there are some security
and portability to device scheduler goals in play) with an infinite number of
registers. A VMFB, which the compiler produces by default is a serialization of
this. The bytecode module has an interpreter for it.

## VM modules

There are multiple module types provided by default in the runtime:

* HAL for interacting with devices
* io_params for external data
* check for testing

There are also Python bindings to create a module object dynamically and define
exports in Python. Since the module interface is just natively defined in C,
IREE also lets you load a .so dynamically which exports a function to create a
new named module (this is what emitc produces for example).

When the tools are creating an `iree_vm_context` for execution, this is
primarily about instantiating modules and adding them. Each module resolved its
imports from the exports of the priors. In practice, an io_params module is
added to access parameter providers, a hal module for devices, one bytecode
module for each vmfb listed, and a native .so module for any .so listed.

That's all it is at the runtime. There's a lot of machinery on the compiler
side for producing these modules and their interplay. The lowest level there to
get a feel for what the compiler can emit, either a bytecode or a C based
export, look at the vm dialect.

## Lowering from the VM to C

There has been talk for years of having a direct lowering of VM to LLVM without
going through C. While desirable in theory, it's just never become a priority...
The C based export is what embedded folks want (because you never want to pin
your system toolchain to a random LLVM like that). And the bytecode side has
never been the bottleneck for others. It's also an annoying, maintenance prone
bit of code to write and just never got done.

The high "it just works" quotient on the bytecode side has probably helped drive
that too. "vmfb" has become a bit synonymous with "IREE" and teams using it
think that is the main thing. But it is just one serialization of a VM module
defining a program... But it has the best tools and debugging experience.

## Call interface

The VM call interface is modeled as a coroutine, and the bytecode interpreter
supports multi task suspend/resume on a single thread. This is used for a lot
of things (i.e. multi batch, async submissive, interfacing to io fabric, etc).
Most of what people think of as "async" in the context of device interactions
comes down to this and the cooperation of the hal module which provides device
based synchronization and scheduling primitives.

The way it is structured, the VM was not designed to block, but it can suspend.

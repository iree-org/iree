# IREE Security Policy

IREE is a compiler and runtime for machine learning models. The compiler turns
models into native code for CPUs and accelerators, and the runtime executes
that code, often with direct access to device drivers. This policy explains how
to report a vulnerability, how the project responds, and which security
properties users can and cannot expect from IREE.

- [Reporting a vulnerability](#reporting-a-vulnerability)
- [How we handle reports](#how-we-handle-reports)
- [Supported versions](#supported-versions)
- [Security model](#security-model)
- [Issues that are not security vulnerabilities](#issues-that-are-not-security-vulnerabilities)
- [Using IREE securely](#using-iree-securely)
- [Build and release security](#build-and-release-security)
- [Feedback on this policy](#feedback-on-this-policy)

## Reporting a vulnerability

**Do not report security vulnerabilities through public GitHub issues, pull
requests, discussions, Discord, or the public mailing lists.**

Report vulnerabilities privately in one of two ways:

- Open a draft advisory through
  [GitHub private vulnerability reporting](https://github.com/iree-org/iree/security/advisories/new)
  (preferred).
- Email the IREE security team at
  [iree-security@lists.lfaidata.foundation](mailto:iree-security@lists.lfaidata.foundation).

Please include:

- A descriptive title and the affected component, for example the compiler,
  the runtime, the Python packages, a HAL driver, the parameter file parsers,
  or the tokenizer.
- The IREE version or commit, and the platform and HAL driver you used.
- A technical description and a minimal reproducer, such as an input program,
  a `.vmfb` file, a parameter file, or tokenizer input.
- Who can exploit the issue and what they gain. Reports that name the trust
  boundary from the [security model](#security-model) that is crossed are
  triaged faster.
- Whether the issue is public or known to third parties, with details.
- Your name and affiliation, if you would like to be credited.

### When to report

- You found a potential vulnerability in IREE.
- You are unsure how a potential vulnerability affects IREE.
- You found a vulnerability in a third-party component that IREE depends on
  or vendors, such as LLVM/MLIR, FlatCC, or Tracy, and IREE is affected. If the
  component is not yet fixed upstream, report it there first, then tell us.
- You found a way to compromise IREE's CI or release infrastructure, or
  evidence that it has been compromised. For example:
  - A workflow that gives pull request code access to secrets.
  - A way to publish to PyPI without a release maintainer approving the
    `pypi` deployment environment.
  - A leaked token or credential, or a compromised account of an `iree-org`
    member.
  - A way to run or persist code on a self-hosted CI runner outside the job
    you were given.
  - A compromised third-party GitHub Action, or a compromised build image from
    [iree-org/base-docker-images](https://github.com/iree-org/base-docker-images).
  - A release artifact on PyPI or GitHub releases that does not match what
    IREE's release process produced.

### When not to report

- You have an automated scanner result that you cannot explain or reproduce.
- You need help using IREE securely or applying security updates. Ask in
  [GitHub Discussions](https://github.com/iree-org/iree/discussions) or on
  [Discord](https://discord.gg/wEWh6Z9nMU) instead.
- The issue is listed under
  [Issues that are not security vulnerabilities](#issues-that-are-not-security-vulnerabilities).
  File it as a
  [regular bug](https://github.com/iree-org/iree/issues/new?template=bug_report.yml).

## How we handle reports

Reports are handled by the IREE security response team: the voting members of
the Technical Steering Committee listed in [CONTRIBUTING.md](CONTRIBUTING.md),
together with any maintainers the TSC brings in for a given report. For each
report we:

1. **Acknowledge** it within 7 days.
2. **Assess** it within 14 days of acknowledging it. We tell the reporter
   whether we accept it as a vulnerability and how severe we rate it (using
   [CVSS](https://www.first.org/cvss/)), or why we decline it.
3. **Fix** it in private, using the advisory's temporary private fork, and
   keep the reporter informed. We may ask the reporter to review the fix.
4. **Disclose** it by publishing a
   [GitHub security advisory](https://github.com/iree-org/iree/security/advisories)
   once a release with the fix is available, and request a CVE ID where
   appropriate.

Every report is either published or closed within 90 days of submission. We may
extend this by agreement with the reporter, for example when the fix depends on
an upstream project. We credit reporters in the advisory unless they prefer to
stay anonymous. We ask reporters not to exploit the issue or disclose it
publicly before the advisory is published.

A report closed as not a security issue may still be a valid bug. Please
re-file it as a regular issue.

## Supported versions

IREE publishes nightly candidate releases (`X.Y.ZrcYYYYMMDD`) and promotes selected release to stable releases (`X.Y.Z`), as
described in [RELEASING.md](RELEASING.md).

| Version                             | Security fixes                         |
| ----------------------------------- | -------------------------------------- |
| `main` and the next nightly release | Yes                                    |
| Latest stable release               | Critical fixes, on a best-effort basis |
| Older releases                      | No                                     |

Security fixes land on `main`, ship in the next nightly release, and are
included in the next stable release. Fixes are not backported to older
releases. For a critical vulnerability the TSC may publish a patch release of
the latest stable release. Upgrading to the latest stable release is the
supported path for every older release.

The deprecated PyPI packages, like, `iree-compiler`, `iree-runtime`,
`iree-runtime-instrumented`, and `iree-tools-xla` receive no fixes. Use
`iree-base-compiler` and `iree-base-runtime` instead.

Tampering with published artifacts, such as an unauthorized change to a
package on PyPI or a file on the GitHub releases page, applies to every
release. Report suspected tampering through the private channels above.

### Verifying release artifacts

IREE's release workflows sign a build provenance attestation with
[Sigstore](https://www.sigstore.dev/) for every file they publish. The
attestation records the workflow, commit, and run that produced the file.
Verification fails if the file was changed afterwards, or if it was produced
by anything other than IREE's release workflows. Files published before IREE
introduced attestations have none.

| Files | Signing workflow |
| --- | --- |
| Nightly candidate release assets on [GitHub releases](https://github.com/iree-org/iree/releases), such as the `iree-base-compiler` and `iree-base-runtime` wheels and the `iree-dist` tarballs | `build_package.yml` |
| Stable wheels on PyPI and on the stable GitHub release | `publish_pypi.yml` |

Verify a downloaded file with the
[GitHub CLI](https://cli.github.com/manual/gh_attestation_verify), using the
signing workflow from the table:

```sh
gh attestation verify <file> \
  --repo iree-org/iree \
  --signer-workflow iree-org/iree/.github/workflows/<signing workflow>
```

Stable wheels are uploaded to PyPI with
[trusted publishing](https://docs.pypi.org/trusted-publishers/), so PyPI also
stores a [PEP 740](https://peps.python.org/pep-0740/) attestation for each file
and shows it on the file's page. Verify it with
[pypi-attestations](https://github.com/pypi/pypi-attestations):

```sh
pypi-attestations verify pypi --repository https://github.com/iree-org/iree \
  pypi:<wheel file name>
```

## Security model

**A compiled IREE module (`.vmfb`) is an executable.** It contains executable code, that the runtime loads into the host process, including device binaries
such as SPIR-V, HSACO, or PTX that run on accelerators.
Loading a module runs that code with the privileges of the host process.
Loading an untrusted `.vmfb` is equivalent to running an untrusted executable
or loading an untrusted shared library with `dlopen`.

The table below lists which inputs IREE treats as trusted, and which inputs it
is designed to handle safely when they come from an untrusted source.

| Input | Trust | What IREE guarantees |
| --- | --- | --- |
| Compiler inputs: MLIR, imported models, and compiler flags | Trusted | Nothing. The compiler is not hardened against malicious input. |
| Compiled modules (`.vmfb`) and the executables they contain | Trusted | Nothing. Modules contain native code that runs in the host process or on a device. |
| Arguments to runtime APIs (C, Python, VM, HAL) | Trusted | Only the argument checks each API documents. |
| Environment variables, flags, flag files, plugins, local files, and caches | Trusted | Nothing. |
| Parameter files (`.irpa`, `.gguf`, `.safetensors`) | May be untrusted | Parsing and indexing a parameter file does not corrupt memory or run code. |
| Tokenizer definitions and the text being encoded or decoded | May be untrusted | Tokenizing and detokenizing does not corrupt memory or run code. |
| Tensor data passed to a compiled function | May be untrusted, within the program's contract | Compiled programs check argument shapes and element types against their signature. Values are not checked. |
| Debugging, tracing, and profiling features, their output, and their network peers | Trusted | Nothing. These features are meant for development on trusted machines. |
| Network peers of collective communication | Trusted | Nothing. Collectives have no authentication. |

A security vulnerability is a bug that lets an attacker who controls only the
untrusted inputs in this table gain a capability they did not already have, for
example memory corruption, code execution, or reading data outside those
inputs. Weaknesses in the build and release pipeline that would allow
published artifacts to be tampered with are also vulnerabilities.

The runtime verifies the structure of bytecode modules when loading them. This
catches corrupted files and compiler bugs early. It does not make an untrusted
module safe to run, because the executables inside it are native code.

## Issues that are not security vulnerabilities

Like most compilers and native libraries, IREE does not validate every input to
every function. The calling code is responsible for passing valid arguments.
An attacker who can already supply a compiled module, run code locally, or
modify local files gains nothing by exploiting IREE. Report the following as
[regular bugs](https://github.com/iree-org/iree/issues/new?template=bug_report.yml),
not as security vulnerabilities:

- **Compiler crashes, hangs, and miscompilation.** Any behavior of
  `iree-compile`, `iree-opt`, the compiler C and Python APIs, or the model
  importers on malformed or malicious input. This includes assertion failures,
  out-of-bounds access, unbounded memory use, and incorrect generated code.
- **Compiler access to local files and tools.** Compiler inputs and flags can
  name files to read and embed (such as executable objects referenced by
  absolute path), directories to write (such as dump and debug output),
  plugins to load, and external tools to run (such as a system linker). This
  is by design.
- **Behavior of compiled modules.** Anything a malicious `.vmfb` can do,
  including running arbitrary native code, accessing memory out of bounds, or
  hanging a device.
- **Invalid use of runtime APIs.** Crashes or out-of-bounds access caused by
  invalid arguments to the C, Python, VM, or HAL APIs, or by calling internal
  APIs or APIs documented as unsafe.
- **Out-of-bounds access driven by tensor values.** Compiled programs do not
  generally bounds-check data-dependent indices, such as gather and scatter
  indices or token IDs. Validating those values is the application's
  responsibility.
- **Numerical differences.** Different results across devices, backends,
  element types, platforms, or versions, loss of precision, and
  non-determinism.
- **Resource consumption.** IREE runs workloads designed to saturate CPUs,
  accelerators, and memory. Inputs that cause large allocations or
  long-running work are expected.
- **Trusted local state.** Issues that require control of environment
  variables (such as `IREE_PY_RUNTIME_FLAGS`), command-line flags, flag files,
  executable plugin libraries (`--executable_plugin=`), or the contents of
  caches and dump directories.
- **Developer tools, samples, and experimental code.** Tools such as
  `iree-run-module` and `iree-benchmark-module`, test utilities, the `samples/`
  and `experimental/` directories, and features documented as experimental are
  not hardened for untrusted environments. A bug in library code that such a
  tool calls is still judged by the [security model](#security-model).
- **Information exposed by debugging features.** Data revealed by tracing,
  profiling, dump, or debug information features, including through their
  network listeners and output files. See
  [Debugging, tracing, and profiling](#debugging-tracing-and-profiling).
- **Unauthenticated collectives.** Data exposed by collective communication
  over an untrusted network.
- **Third-party issues that IREE does not reach.** Report these to the
  upstream project.

If your advisory is closed for one of these reasons, please don't be
discouraged. These reports are still valuable, and we encourage you to re-file
them as regular issues so they can be tracked and fixed.

## Using IREE securely

### Untrusted models

Treat compiled modules like executables and model sources like source code.

- **Only load `.vmfb` files from sources you trust**, and verify their
  provenance, for example with checksums published by their author.
- **Prefer compiling programs yourself and loading weights separately.**
  Parameter files contain data, not code, and are the safest way to use
  weights from a third party. A program you compiled from a model source you
  trust, combined with third-party weights in a parameter file, exposes far
  less than a third-party `.vmfb`.
- **Isolate untrusted work.** Compile untrusted model sources, and run modules
  you cannot fully trust, in a sandbox such as a container or virtual machine
  without access to sensitive files, credentials, or networks. Giving a sandbox
  access to GPUs or other accelerators widens its attack surface through the
  device drivers.
- **Remember that importers run upstream code.** Importing a model from
  PyTorch (through iree-turbine), ONNX, or TensorFlow Lite uses those
  frameworks to load it and inherits their security properties. A PyTorch
  model is Python code; see the
  [PyTorch security policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md).

Trust in a model is not binary. Choose the level of caution that matches where
the model came from and how you deploy it.

### Untrusted inputs

If a compiled program processes input from untrusted users, for example in an
inference service:

- Validate shapes, element types, and sizes before calling the program, and
  limit them to what the program was compiled and tested for.
- Validate data-dependent indices, such as token IDs against the vocabulary
  size, before they reach gather, scatter, or embedding operations.
- Run the service with the least privileges it needs, and keep IREE up to
  date.
- Address threats that live in the model itself, such as prompt injection or
  adversarial inputs, in your application. They are outside IREE's scope.

### Multi-tenant deployments

IREE does not isolate workloads that share a process or a device from each
other. Separate tenants with operating system processes, virtual machines, or
dedicated devices, as your threat model requires.

### Debugging, tracing, and profiling

IREE's debugging, tracing, and profiling features exist to show developers what
a workload is doing, so they expose information about it: program structure,
tensor contents, source paths, memory addresses, and timing. They are meant for
development on trusted machines and are not hardened. Treat their output as
being as sensitive as the model and the data it processes.
Do not enable them in production or with sensitive data unless you control where their output goes.

### Multi-device and distributed execution

Collective communication channels, such as those built on NCCL or RCCL, are
meant for trusted peers on trusted networks. IREE adds no authentication or
encryption to them. Follow the security guidance of the underlying
communication library.

## Build and release security

_Audience_: contributors and reviewers, especially when changing files under
`.github/` or `build_tools/`.

IREE's CI aims to keep its pipelines open and reproducible while protecting
the infrastructure and the artifacts the project publishes. Changes to
workflows and build scripts must preserve these principles:

- **Code from pull requests runs without release credentials.** Workflows
  that build or test pull request code must not have access to release
  tokens. Workflows triggered by `pull_request_target` must never check out or
  run code from the pull request.
- **Self-hosted runners are not trusted with release credentials.** Some CI
  jobs, such as GPU tests, run on persistent self-hosted runners shared by
  unrelated pull requests. Do not give those jobs release credentials.
- **CI artifacts are for testing only.** CI builds use a shared remote
  compiler cache, so their outputs could be affected by a poisoned cache. Do
  not publish CI artifacts or run them where sensitive data is available. They
  expire without notice.
- **Third-party actions are pinned.** GitHub Actions are pinned to full commit
  SHAs and updated through Dependabot.
- **Build images are pinned.** CI and release builds reference container
  images by digest (`@sha256:...`), not by tag, whether the image comes from
  [iree-org/base-docker-images](https://github.com/iree-org/base-docker-images)
  or from an upstream project.
- **Release builds start clean.** Release packages are built from a tagged
  commit on GitHub-hosted runners, without shared build caches.
- **Published files are attested.** The release workflows sign a build
  provenance attestation with `actions/attest` for every file they upload to
  GitHub releases or PyPI. New release files must be attested in the job that
  produces them, before they are uploaded.
- **Stable releases are promoted, not rebuilt.** A stable release republishes
  the wheels of a tested nightly release with only the version changed.
- **PyPI uploads use trusted publishing.** The `publish_pypi.yml` workflow
  publishes stable releases with
  [trusted publishing](https://docs.pypi.org/trusted-publishers/). No PyPI
  credentials are stored in GitHub: PyPI exchanges the publishing job's OIDC
  token for a short-lived upload token.
- **Publishing needs approval.** Publishing jobs run in the protected `pypi`
  environment, which only a release maintainer can approve and which only
  accepts runs from `main`. These jobs have no permissions besides
  `id-token: write`, and they only upload wheels that an earlier job promoted,
  checked, and attested (see the
  [release management guide](https://iree.dev/developers/general/release-management/)).
- **Two-factor authentication is required.** The `iree-org` GitHub
  organization requires two-factor authentication. GitHub removes members and
  outside collaborators who do not enable it, including bot accounts.

A suspected compromise of CI, credentials, or release artifacts is a security
vulnerability. Report it privately as described in
[Reporting a vulnerability](#reporting-a-vulnerability), not in a public issue
or pull request. When a credential may have leaked, the TSC revokes and rotates
it first and investigates afterwards. When a published artifact may have been
tampered with, the TSC removes it from PyPI and GitHub releases and publishes
an advisory naming the affected versions.

## Feedback on this policy

To suggest improvements to this policy, open an issue or a pull request. **Do
not** report potential vulnerabilities through a pull request.

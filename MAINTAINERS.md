# IREE Maintainer Information

The IREE project operates through collaborative development via
Discord, Pull Requests, and the mailing list. While much day to day work
can take place without much fanfare, the variety of code in the project
is large, and this page attempts to document "go to" people with specialist
skills, responsibility and insights for certain components. See also
[CODEOWNERS](.github/CODEOWNERS) for path-based reviewers for various
components. If in doubt and there is a specific CODEOWNER for the path you
are working on, consider that to be a more authoritative source than this file.
This file attempts to outline long term responsibility for questions of 
evolution, health, and design.

This file is kept in the `iree` core repository but can refer to other
affiliated repositories at need. This is meant to help "direct traffic" and
individual projects should be authoritative about their status.

## Overall

Stella Laurenzo (@stellaraccident) is the maintainer of last resort for
uncovered components, questions of project direction, etc.

## Compiler Maintainers

* Runtime Interface: Ben Vanik (@benvanik)
* High Level Optimizations: Mahesh Ravishankar (@MaheshRavishankar)
* Code Generation: Mahesh Ravishankar (@MaheshRavishankar)
* Input Pipelines:

  * Torch: Rob Suderman (@rsuderman)
  * TOSA: Maintainer Needed
  * StableHLO: Maintainer Needed

## Runtime Maintainers

* Overall: Ben Vanik (@benvanik)
* Default HAL Drivers: Lei Zhang (@antiagainst)

## Build Tools, Infra, and Dependencies

* LLVM Dependency: Mahesh Ravishankar (@MaheshRavishankar)
* CI, Docs, and Tools: Scott Todd (@ScottTodd)
* Other Dependencies: Ben Vanik (@benvanik) and Scott Todd (@ScottTodd)

## APIs

* Compiler C API: Stella Laurenzo (@stellaraccident)
* Python Bindings: Stella Laurenzo (@stellaraccident)
* Turbine: Stella Laurenzo (@stellaraccident)
* PJRT: Maintainer Needed

## Releasing

* Python Releases: Stella Laurenzo (@stellaraccident)
* Other packages, nightlies, and infra: Maintainer Needed


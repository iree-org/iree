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

## Technical Steering Committee

The Technical Steering Committee (TSC) is responsible for the overall
technical direction and health of the project, as set out in the
[IREE Technical Charter](https://github.com/lfai/foundation/blob/main/technical%20project%20charters/IREE%20Technical%20Charter%20Final%205-14-2024.docx.pdf),
and represents the project to the LF AI & Data Foundation's Technical
Advisory Council (TAC).

The current TSC voting members are listed in
[CONTRIBUTING.md](CONTRIBUTING.md), as provided for in section 2.b of
the charter.

### Operation

* The TSC decides by lazy consensus where possible. When a formal vote
  is needed, each voting member has one vote and voting follows the
  rules in section 3 of the Technical Charter: majority of those
  present at a quorate meeting, or majority of all voting members for
  electronic votes. License exceptions and charter amendments require
  a two-thirds vote of the entire TSC (charter sections 7.c and 8.a).
* The chairperson is elected by the TSC voting members and serves as
  the project's representative to the LF AI & Data TAC.
* Changes to TSC membership (additions, removals, emeritus status) are
  decided by TSC vote.
* The TSC meets monthly. Meetings are open to the public and announced
  on the [iree-technical-discussion mailing list](https://lists.lfaidata.foundation/g/iree-technical-discussion)
  and Discord.

## Overall

Questions of project direction or components without a listed
maintainer are escalated to the
[Technical Steering Committee](#technical-steering-committee).

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

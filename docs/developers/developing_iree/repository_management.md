# IREE Repository Management

Due to the process by which we synchronize this GitHub project with our internal
Google source code repository, there are some oddities in our workflows and
processes. We aim to minimize these, and especially to minimize their impact on
external contributors, but they are documented here for clarity and
transparency. If any of these things are particularly troublesome or painful for
your workflow, please reach out to us so we can prioritize a fix.

## Branches

The default branch is called `main`. PRs should be sent there. We also have a
`google` branch that is synced from the Google internal source repository. This
branch is merged from and to `main` frequently to upstream any changes coming
from `google` and as part of updating the internal source repository. For the
most part this integration with `google` should have minimal effect on normal
developer workflows.

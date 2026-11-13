# Nexus Router gateway launcher

Requires Python 3.11+ and Node 20+. The npm launcher installs the matching Python
release in a private environment and forwards arguments without a shell.

After publication: `npx @ghabs/nexus-router-gateway init`.

Before publication, pack this directory with `npm pack`, install the Python wheel,
and set `NEXUS_GATEWAY_EXECUTABLE` to its `nexus-router-gateway` executable. Or set
`NEXUS_GATEWAY_WHEEL` to a trusted local wheel to bootstrap it with its dependencies.
No package from this repository has been published by creating these files.

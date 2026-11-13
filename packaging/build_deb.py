"""Build a Debian binary package using stdlib (no root/dpkg required).

Runtime dependencies come from apt; this does not bundle Google credentials.
"""

import io
from pathlib import Path
import tarfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def tar(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, content, mode in files:
            entry = tarfile.TarInfo(path)
            entry.size, entry.mode, entry.mtime = len(content), mode, 0
            archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()


def build():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    name, version = project["name"], project["version"]
    metadata = (f"Package: {name}\nVersion: {version}\nArchitecture: all\n"
                "Maintainer: Nexus Router maintainers\nSection: net\nPriority: optional\n"
                "Depends: python3 (>= 3.11), python3-httpx (>= 0.27), python3-yaml, python3-google-auth, "
                "python3-requests, python3-jwt (>= 2.8), python3-cryptography\n"
                "Description: Claude Cowork gateway with Vertex AI model and effort routing\n")
    data = [(f"usr/lib/python3/dist-packages/nexus_router/{path.name}", path.read_bytes(), 0o644)
            for path in (ROOT / "src").iterdir() if path.suffix in {".py", ".sql", ".yaml", ".html", ".js", ".css"}]
    data += [("usr/bin/nexus-router-gateway", b"#!/usr/bin/python3\nfrom nexus_router.gateway_cli import main\nraise SystemExit(main())\n", 0o755),
             ("usr/lib/systemd/system/nexus-router-gateway.service", (ROOT / "packaging/nexus-router-gateway.service").read_bytes(), 0o644),
             ("usr/share/doc/nexus-router-gateway/gateway.example.yaml", (ROOT / "src/gateway.example.yaml").read_bytes(), 0o644)]
    output = ROOT / "dist" / f"{name}_{version}_all.deb"
    output.parent.mkdir(exist_ok=True)
    with output.open("wb") as file:
        file.write(b"!<arch>\n")
        for member, content in [("debian-binary", b"2.0\n"), ("control.tar.gz", tar([("control", metadata.encode(), 0o644)])), ("data.tar.gz", tar(data))]:
            header = f"{member:<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(content):<10}`\n"
            file.write(header.encode())
            file.write(content)
            if len(content) % 2:
                file.write(b"\n")
    print(output)


if __name__ == "__main__":
    build()

"""Render a Homebrew formula from a built source archive and its release URL."""

import argparse
import hashlib
from pathlib import Path
import tomllib
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--url", required=True, help="Published HTTPS URL or file:// URL for a local smoke test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if urlparse(args.url).scheme not in {"https", "file"}:
        parser.error("Use HTTPS for a release, or file:// for a local archive")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    if '"' in args.url or "\n" in args.url:
        parser.error("Invalid source URL")
    formula = f'''class NexusRouterGateway < Formula
  desc "Claude Cowork gateway with Vertex model and effort routing"
  url "{args.url}"
  version "{project['version']}"
  sha256 "{hashlib.sha256(args.archive.read_bytes()).hexdigest()}"
  depends_on "python@3.12"

  def install
    system Formula["python@3.12"].opt_bin/"python3.12", "-m", "venv", libexec
    # A published tap can vendor pinned resources for fully offline builds.
    system libexec/"bin/python", "-m", "pip", "install", buildpath
    bin.install_symlink libexec/"bin/nexus-router-gateway"
  end

  service do
    run [opt_bin/"nexus-router-gateway", "serve"]
    keep_alive true
    log_path var/"log/nexus-router-gateway.log"
    error_log_path var/"log/nexus-router-gateway.log"
  end

  test do
    assert_match "nexus-router-gateway", shell_output("#{{bin}}/nexus-router-gateway --version")
  end
end
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formula)
    print(args.output)


if __name__ == "__main__":
    main()

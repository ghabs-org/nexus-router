"""Installable entry point for the standalone gateway and Cowork setup files."""

import argparse
import json
import logging
import os
from pathlib import Path
import secrets
import sys


def _private_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as file:
        file.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nexus-router-gateway")
    parser.add_argument("--version", action="version", version="nexus-router-gateway 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)
    default_config = Path.home() / ".config/nexus-router/gateway.yaml"
    default_state = Path.home() / ".local/state/nexus-router"
    for name in ("init", "check", "serve", "enable-provider", "cowork-config", "codex-config", "antigravity-config", "mcp", "reconcile-budget"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=default_config)
        command.add_argument("--state-dir", type=Path, default=default_state)
        command.add_argument("--env-file", type=Path)
        if name == "init":
            command.add_argument("--template", choices=["cowork", "codex"], default="cowork")
        if name == "reconcile-budget":
            command.add_argument("--request-id", required=True)
            command.add_argument("--actual-cost-usd", required=True, type=float)
            command.add_argument("--note", required=True)
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", type=int, default=7772)
        if name == "enable-provider":
            command.add_argument("provider")
        if name in {"cowork-config", "codex-config", "antigravity-config", "mcp"}:
            command.add_argument("--url", default="http://127.0.0.1:7772")
            command.add_argument("--token-env", default="NEXUS_GATEWAY_TOKEN")
            if name != "mcp":
                command.add_argument("--output", type=Path, required=True)
            command.add_argument("--oidc-client-id")
    args = parser.parse_args(argv)
    os.environ["NEXUS_ROUTER_STATE_ROOT"] = str(args.state_dir.resolve())
    # Set before importing the DB/paths modules, including in installed wheels.
    os.environ["NEXUS_ROUTER_DB_PATH"] = str(args.state_dir.resolve() / "data/router.sqlite")
    try:
        if args.command == "reconcile-budget":
            from . import db
            db.DB_PATH = args.state_dir.resolve() / "data/router.sqlite"
            from .gateway_budget import BudgetLedger
            db.ensure_schema()
            BudgetLedger().reconcile(args.request_id, args.actual_cost_usd, args.note)
            print("Budget reservation reconciled with an audit entry")
            return 0
        if args.command == "mcp":
            from .gateway_mcp import serve
            serve(args.url, args.token_env)
            return 0
        if args.command in {"codex-config", "antigravity-config"}:
            from urllib.parse import urlparse
            import re
            url = urlparse(args.url)
            if url.scheme != "https" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("Remote gateway requires HTTPS")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.token_env):
                raise ValueError("Invalid token environment variable")
            if args.command == "codex-config":
                text = ('model = "nexus-auto"\nmodel_provider = "nexus"\n'
                        '[model_providers.nexus]\nname = "Nexus Router"\nwire_api = "responses"\n'
                        'base_url = ' + json.dumps(args.url.rstrip("/") + "/v1") + '\n'
                        'env_key = ' + json.dumps(args.token_env) + '\n')
            else:
                text = json.dumps({"mcpServers": {"nexus-router": {"command": "nexus-router-gateway",
                                  "args": ["mcp", "--url", args.url, "--token-env", args.token_env]}}}, indent=2) + "\n"
            _private_write(args.output, text)
            print(f"Wrote {args.output}; merge into the client configuration. No existing settings changed.")
            return 0
        if args.command == "init":
            from importlib.resources import files
            template = files(__package__).joinpath("gateway.codex.yaml" if args.template == "codex" else "gateway.example.yaml").read_text()
            env_path = args.env_file or args.config.with_suffix(".env")
            if args.config.exists() or env_path.exists():
                raise ValueError("Configuration or credentials already exist; refusing to overwrite")
            _private_write(env_path, f"NEXUS_GATEWAY_TOKEN={secrets.token_urlsafe(36)}\n")
            _private_write(args.config, template)
            print(f"Created {args.config} and {env_path}. " + ("Set OPENAI_API_KEY in the gateway process." if args.template == "codex" else "Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION before serving."))
            return 0
        env_path = args.env_file or args.config.with_suffix(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                key, separator, value = line.partition("=")
                if not separator or not key.strip().replace("_", "").isalnum():
                    raise ValueError("Invalid environment file; use KEY=value without shell expressions")
                os.environ.setdefault(key.strip(), value.strip())
        from . import db
        # Also supports repeated invocations in embedding processes/tests.
        db.DB_PATH = args.state_dir.resolve() / "data/router.sqlite"
        from .gateway import GatewayPolicy, GatewayAuth, create_server, load_config
        config = load_config(args.config)
        policy = GatewayPolicy(config)
        if args.command == "cowork-config":
            from urllib.parse import urlparse
            url = urlparse(args.url)
            if url.scheme != "https" and url.hostname not in {"127.0.0.1", "localhost", "::1"}:
                raise ValueError("Shared gateways require an HTTPS URL")
            output = {"inferenceProvider": "gateway", "inferenceGatewayBaseUrl": args.url.rstrip("/"),
                      "inferenceGatewayAuthScheme": "bearer", "modelDiscoveryEnabled": False,
                      "inferenceModels": [{"name": policy.alias, "labelOverride": "Automatic (Nexus)",
                                           "behavesAs": config["behaves_as"]}] + [
                          {"name": target.model, "anthropicFamilyTier": policy.models[label]["tier"]}
                          for label, target in policy.targets.items()],
                      "autoModeClassifierModel": config["behaves_as"],
                      "alwaysStartWithDefaultModel": True, "defaultModelEffort": "medium",
                      "inferenceGatewayAutoModeServerChecks": False}
            if args.oidc_client_id:
                oidc = config.get("oidc") or {}
                if not oidc.get("issuer"):
                    raise ValueError("Configure the gateway OIDC issuer before generating SSO settings")
                output.update({"inferenceCredentialKind": "interactive",
                               "inferenceGatewayOidc": {"clientId": args.oidc_client_id, "issuer": oidc["issuer"]}})
            else:
                clients = list(config.get("clients", {}).values())
                if not clients or not os.environ.get(clients[0]["token_env"]):
                    raise ValueError("Client token is unavailable; supply the managed environment file")
                output["inferenceGatewayApiKey"] = os.environ[clients[0]["token_env"]]
            _private_write(args.output, json.dumps(output, indent=2) + "\n")
            print(f"Wrote {args.output}. Import these settings through Claude Desktop's enterprise configuration; API-key output contains a secret.")
            return 0
        if args.command == "enable-provider":
            if args.provider not in config["providers"]:
                raise ValueError("Provider is absent from gateway configuration")
            db.ensure_schema()
            with db._connect() as conn:
                conn.execute("INSERT INTO providers (provider,enabled,status) VALUES (?,1,'enabled') "
                             "ON CONFLICT(provider) DO UPDATE SET enabled=1,status='enabled'", (args.provider,))
            print(f"Enabled provider {args.provider} in the router database")
            return 0
        GatewayAuth(config)
        if args.command == "check":
            print(f"Configuration valid: {len(policy.models)} models, {len(policy.profiles)} profiles. No upstream request made.")
            return 0
        logging.basicConfig(level=logging.INFO)
        server = create_server(config, args.host, args.port)
        print(f"Nexus gateway listening on {args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    except Exception as exc:
        # Configuration failures are actionable; upstream credential objects and
        # full request bodies must never be rendered by a top-level traceback.
        if isinstance(exc, (ValueError, FileNotFoundError, FileExistsError)):
            print(f"Gateway setup error: {exc}", file=sys.stderr)
        else:
            print(f"Gateway setup failed ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

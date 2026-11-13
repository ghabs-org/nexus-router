"""Authenticate gateway clients independently of upstream Google credentials."""

import hashlib
import hmac
import os
from urllib.parse import urlparse

from .gateway_policy import PolicyError


class GatewayAuth:
    def __init__(self, config):
        self.clients = []
        seen = set()
        for name, client in config.get("clients", {}).items():
            secret = os.environ.get(client["token_env"], "")
            if not secret:
                raise ValueError(f"Client credential environment variable is missing: {client['token_env']}")
            if secret in seen:
                raise ValueError("Different gateway clients must not share the same credential")
            seen.add(secret)
            self.clients.append((name, secret, client.get("limits", {})))
        self.oidc = config.get("oidc")
        self.jwks = None
        if self.oidc:
            import jwt

            for key in ("issuer", "jwks_url"):
                if urlparse(self.oidc[key]).scheme != "https":
                    raise ValueError(f"OIDC {key} must use HTTPS")
            if not self.oidc.get("audience"):
                raise ValueError("OIDC audience is required")
            self.jwks = jwt.PyJWKClient(self.oidc["jwks_url"], cache_keys=False, lifespan=300, timeout=5)
        if not self.clients and not self.oidc:
            raise ValueError("Configure at least one client credential or an OIDC issuer")

    def authenticate(self, headers):
        bearer = headers.get("Authorization", "")
        token = bearer[7:].strip() if bearer.lower().startswith("bearer ") else headers.get("x-api-key", "")
        if not token or len(token) > 16384:
            raise PolicyError("Gateway authentication required", 401)
        identity, limits = None, None
        for name, expected, client_limits in self.clients:
            if hmac.compare_digest(token.encode(), expected.encode()):
                identity, limits = f"key:{name}", client_limits
        if identity is None and self.oidc:
            import jwt

            try:
                key = self.jwks.get_signing_key_from_jwt(token).key
                claims = jwt.decode(token, key, algorithms=["RS256", "ES256"],
                                    audience=self.oidc["audience"], issuer=self.oidc["issuer"],
                                    options={"require": ["exp", "iss", "sub", "aud"]})
                allowed = self.oidc.get("allowed_subjects")
                if allowed and claims["sub"] not in allowed:
                    raise PolicyError("Gateway access denied", 403)
                identity = f"oidc:{claims['iss']}:{claims['sub']}"
                limits = self.oidc.get("limits", {})
            except PolicyError:
                raise
            except Exception:
                raise PolicyError("Gateway credential is invalid or expired", 401) from None
        if identity is None:
            raise PolicyError("Gateway credential is invalid", 401)
        return hashlib.sha256(identity.encode()).hexdigest(), limits

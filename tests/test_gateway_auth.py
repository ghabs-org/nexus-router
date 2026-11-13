"""Cryptographic OIDC and ADC refresh checks; no real identity provider is called."""

import json
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric import rsa

from src.gateway_auth import GatewayAuth
from src.gateway_policy import PolicyError
from src.vertex import VertexCredentials


def test_oidc_verifies_signature_issuer_audience_expiry_subject_and_rotation(monkeypatch):
    auth = GatewayAuth({"oidc": {"issuer": "https://identity.example/tenant", "audience": "gateway-app",
                                 "jwks_url": "https://identity.example/keys"}})
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update(kid="first", use="sig", alg="RS256")
    monkeypatch.setattr(auth.jwks, "fetch_data", lambda: {"keys": [jwk]})
    claims = {"iss": "https://identity.example/tenant", "aud": "gateway-app", "sub": "alice", "exp": int(time.time()) + 60}
    def token(values, signing_key=key, kid="first"):
        return jwt.encode(values, signing_key, algorithm="RS256", headers={"kid": kid})
    principal, _ = auth.authenticate({"Authorization": f"Bearer {token(claims)}"})
    assert len(principal) == 64 and "alice" not in principal
    for invalid in ({**claims, "iss": "https://wrong.example"}, {**claims, "aud": "other-app"},
                    {**claims, "exp": int(time.time()) - 60}, {k: v for k, v in claims.items() if k != "sub"}):
        with pytest.raises(PolicyError, match="invalid or expired"):
            auth.authenticate({"Authorization": f"Bearer {token(invalid)}"})
    wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(PolicyError):
        auth.authenticate({"Authorization": f"Bearer {token(claims, wrong_key)}"})
    second = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(wrong_key.public_key()))
    second.update(kid="second", use="sig", alg="RS256")
    monkeypatch.setattr(auth.jwks, "fetch_data", lambda: {"keys": [second]})
    assert auth.authenticate({"Authorization": f"Bearer {token(claims, wrong_key, 'second')}"})[0] == principal


def test_adc_refresh_is_shared_and_thread_safe(monkeypatch):
    google_auth = pytest.importorskip("google.auth")
    from google.auth.credentials import Credentials

    class TestCredentials(Credentials):
        def __init__(self):
            super().__init__()
            self.refreshes = 0

        def refresh(self, request):
            self.refreshes += 1
            self.token = f"access-{self.refreshes}"

    credentials = TestCredentials()
    calls = []
    def default(**kwargs):
        calls.append(kwargs)
        return credentials, "test-project"
    monkeypatch.setattr(google_auth, "default", default)
    manager = VertexCredentials()
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert list(workers.map(lambda _: manager.token(), range(16))) == ["access-1"] * 16
    assert len(calls) == 1 and credentials.refreshes == 1
    assert calls[0]["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    credentials.token = None
    assert manager.token() == "access-2"

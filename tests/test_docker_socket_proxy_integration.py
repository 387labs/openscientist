"""R6 integration test: exercise the real docker-socket-proxy whitelist.

Stands up tecnativa/docker-socket-proxy with the exact environment from
docker-compose.yml, points a Docker client at it, and asserts that the
container-per-job lifecycle ops are allowed while everything outside the
whitelist is denied (HTTP 403). Skips when Docker is unavailable.
"""

from __future__ import annotations

import time
from contextlib import suppress

import pytest

import docker

PROXY_IMAGE = "tecnativa/docker-socket-proxy:0.3.0"
LIFECYCLE_IMAGE = "hello-world"
_SOCKET = "/var/run/docker.sock"

# Mirror docker-compose.yml exactly.
PROXY_ENV = {
    "CONTAINERS": "1",
    "IMAGES": "1",
    "POST": "1",
    "DELETE": "1",
    "PING": "1",
    "VERSION": "1",
    "INFO": "0",
    "EXEC": "0",
    "NETWORKS": "0",
    "VOLUMES": "0",
    "BUILD": "0",
    "COMMIT": "0",
    "AUTH": "0",
    "SYSTEM": "0",
    "SWARM": "0",
    "SECRETS": "0",
    "CONFIGS": "0",
    "SERVICES": "0",
    "TASKS": "0",
    "NODES": "0",
    "PLUGINS": "0",
    "DISTRIBUTION": "0",
    "SESSION": "0",
    "EVENTS": "0",
}


@pytest.mark.integration
class TestDockerSocketProxy:
    """Behavioural checks against a live socket proxy."""

    @pytest.fixture(scope="class")
    def host_docker(self):
        """Real Docker client on the host, or skip when Docker is unavailable."""
        client = docker.from_env()
        try:
            client.ping()
        except Exception:
            client.close()
            pytest.skip("Docker not available")
        try:
            yield client
        finally:
            client.close()

    @pytest.fixture(scope="class")
    def proxied_client(self, host_docker):
        """A Docker client that reaches the daemon ONLY through the proxy."""
        # Pull via the host (not the proxy) so the lifecycle test does not depend
        # on the proxy's image-pull path; skip the whole class on network issues.
        try:
            host_docker.images.pull(PROXY_IMAGE)
            host_docker.images.pull(LIFECYCLE_IMAGE)
        except Exception as exc:
            pytest.skip(f"could not pull test images: {exc}")

        try:
            proxy = host_docker.containers.run(
                PROXY_IMAGE,
                detach=True,
                environment=PROXY_ENV,
                volumes={_SOCKET: {"bind": _SOCKET, "mode": "ro"}},
                ports={"2375/tcp": ("127.0.0.1", 0)},
            )
        except Exception as exc:
            pytest.skip(f"could not start socket proxy: {exc}")

        try:
            proxy.reload()
            host_port = proxy.attrs["NetworkSettings"]["Ports"]["2375/tcp"][0]["HostPort"]
            client = docker.DockerClient(base_url=f"tcp://127.0.0.1:{host_port}")
            _wait_ready(client)
            try:
                yield client
            finally:
                client.close()
        finally:
            with suppress(Exception):
                proxy.remove(force=True)

    # -- allowed: the job-container lifecycle --------------------------------

    def test_version_and_ping_allowed(self, proxied_client):
        assert proxied_client.version()["Version"]
        assert proxied_client.ping() is True

    def test_list_containers_and_images_allowed(self, proxied_client):
        proxied_client.containers.list(all=True)
        proxied_client.images.list()

    def test_container_lifecycle_allowed(self, proxied_client):
        """create -> start -> wait -> logs -> remove, all through the proxy."""
        container = proxied_client.containers.run(LIFECYCLE_IMAGE, detach=True)
        try:
            container.wait(timeout=60)
            assert b"Hello from Docker" in container.logs()
        finally:
            container.remove(force=True)
        with pytest.raises(docker.errors.NotFound):
            proxied_client.containers.get(container.id)

    # -- denied: everything outside the whitelist ----------------------------

    def test_info_denied(self, proxied_client):
        _assert_forbidden(lambda: proxied_client.info())

    def test_system_df_denied(self, proxied_client):
        _assert_forbidden(lambda: proxied_client.df())

    def test_volumes_denied(self, proxied_client):
        _assert_forbidden(lambda: proxied_client.volumes.list())

    def test_network_create_denied(self, proxied_client):
        _assert_forbidden(lambda: proxied_client.networks.create("r6-denied"))


def _wait_ready(client: docker.DockerClient, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.version()
            return
        except Exception:
            if time.monotonic() > deadline:
                pytest.skip("socket proxy did not become ready")
            time.sleep(0.5)


def _assert_forbidden(call) -> None:
    with pytest.raises(docker.errors.APIError) as exc:
        call()
    assert exc.value.response is not None
    assert exc.value.response.status_code == 403

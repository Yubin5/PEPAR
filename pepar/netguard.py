"""URL fetching that refuses private addresses (SSRF guard), follows redirects by hand, and caps the size."""

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import NameResolutionError, NewConnectionError

from errors import PeparError

MAX_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 10
PRIVATE_MESSAGE = "내부 주소는 번역할 수 없습니다. 허용하려면 config.toml의 allow_private_urls를 켜세요."


def is_public(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%")[0])
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global


def resolve_public(host: str, port: int) -> list[tuple]:
    """Resolved socket addresses for host, or PeparError if any of them is not public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise PeparError("주소를 찾을 수 없습니다.") from e
    if not infos or not all(is_public(i[4][0]) for i in infos):
        raise PeparError(PRIVATE_MESSAGE)
    return infos


def check_url(url: str):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise PeparError("http(s) 주소만 번역할 수 있습니다.")
    resolve_public(p.hostname, p.port or (443 if p.scheme == "https" else 80))


class _Guarded:
    """Connect only to addresses that were just checked, so DNS cannot change between check and connect."""

    def _new_conn(self):
        try:
            infos = resolve_public(self._dns_host, self.port)
        except PeparError as e:
            raise NewConnectionError(self, str(e)) from e
        except OSError as e:
            raise NameResolutionError(self.host, self, e) from e
        err = None
        for family, _, _, _, addr in infos:
            try:
                return socket.create_connection(addr[:2], self.timeout, source_address=self.source_address)
            except OSError as e:
                err = e
        raise NewConnectionError(self, f"Failed to establish a new connection: {err}")


class _HTTPConnection(_Guarded, HTTPConnection):
    pass


class _HTTPSConnection(_Guarded, HTTPSConnection):
    pass


class _HTTPPool(HTTPConnectionPool):
    ConnectionCls = _HTTPConnection


class _HTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _HTTPSConnection


class GuardAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _HTTPPool, "https": _HTTPSPool}


def fetch(url: str, headers: dict, allow_private: bool, timeout: int = 60) -> requests.Response:
    """GET url. Unless allow_private, every hop (including redirects) must resolve to public addresses."""
    session = requests.Session()
    if not allow_private:
        session.trust_env = False
        session.mount("http://", GuardAdapter())
        session.mount("https://", GuardAdapter())
    for _ in range(MAX_REDIRECTS + 1):
        if not allow_private:
            check_url(url)
        r = session.get(url, headers=headers, timeout=timeout, allow_redirects=False, stream=True)
        location = r.headers.get("location")
        if r.status_code in (301, 302, 303, 307, 308) and location:
            r.close()
            url = urljoin(url, location)
            continue
        chunks, size = [], 0
        for chunk in r.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                r.close()
                raise PeparError(f"페이지가 너무 큽니다 (최대 {MAX_BYTES // 1024 // 1024}MB).")
            chunks.append(chunk)
        r._content = b"".join(chunks)
        return r
    raise requests.exceptions.TooManyRedirects()

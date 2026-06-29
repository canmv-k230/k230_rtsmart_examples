#!/usr/bin/env python3
"""Small host-side websocket tester for the RT-Smart libwebsockets samples.

Examples:
  PC as server, device runs ws_client:
    python3 host_ws_test.py server --host 0.0.0.0 --port 7681
    ws_client <pc-ip> 7681

  PC as client, device runs ws_server:
    ws_server 7681
    python3 host_ws_test.py client <device-ip> --port 7681 --message hello
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import socket
import struct
from typing import Dict, Tuple

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
DEFAULT_PORT = 7681
DEFAULT_PATH = "/"
DEFAULT_PROTOCOL = "echo-protocol"

OP_CONTINUATION = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("connection closed")
        data.extend(chunk)
    return bytes(data)


def read_http_headers(sock: socket.socket) -> bytes:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("connection closed during HTTP handshake")
        data.extend(chunk)
        if len(data) > 16384:
            raise ValueError("HTTP headers too large")
    return bytes(data)


def parse_http_headers(raw: bytes) -> Tuple[str, Dict[str, str]]:
    text = raw.decode("iso-8859-1")
    lines = text.split("\r\n")
    start = lines[0]
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            break
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    return start, headers


def websocket_accept(key: str) -> str:
    digest = hashlib.sha1((key + GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def send_frame(sock: socket.socket, opcode: int, payload: bytes = b"", *, mask: bool = False) -> None:
    first = 0x80 | opcode
    length = len(payload)
    header = bytearray([first])

    mask_bit = 0x80 if mask else 0
    if length < 126:
        header.append(mask_bit | length)
    elif length <= 0xFFFF:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", length))

    if mask:
        key = os.urandom(4)
        header.extend(key)
        payload = bytes(byte ^ key[index & 3] for index, byte in enumerate(payload))

    sock.sendall(bytes(header) + payload)


def read_frame(sock: socket.socket) -> Tuple[int, bytes]:
    first, second = recv_exact(sock, 2)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    length = second & 0x7F

    if length == 126:
        length = struct.unpack("!H", recv_exact(sock, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", recv_exact(sock, 8))[0]

    mask_key = recv_exact(sock, 4) if masked else b""
    payload = recv_exact(sock, length) if length else b""

    if masked:
        payload = bytes(byte ^ mask_key[index & 3] for index, byte in enumerate(payload))

    return opcode, payload


def server_handshake(conn: socket.socket, path: str, protocol: str) -> None:
    raw = read_http_headers(conn)
    start, headers = parse_http_headers(raw)
    parts = start.split()
    if len(parts) < 2 or parts[0] != "GET":
        raise ValueError(f"unsupported request line: {start}")
    if parts[1] != path:
        raise ValueError(f"unexpected path {parts[1]!r}, expected {path!r}")

    key = headers.get("sec-websocket-key")
    if not key:
        raise ValueError("missing Sec-WebSocket-Key")

    offered = [item.strip() for item in headers.get("sec-websocket-protocol", "").split(",")]
    selected_protocol = protocol if protocol in offered else None

    response = [
        "HTTP/1.1 101 Switching Protocols",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Accept: {websocket_accept(key)}",
    ]
    if selected_protocol:
        response.append(f"Sec-WebSocket-Protocol: {selected_protocol}")
    response.extend(["", ""])
    conn.sendall("\r\n".join(response).encode("ascii"))


def client_handshake(sock: socket.socket, host: str, port: int, path: str, protocol: str) -> None:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        f"Sec-WebSocket-Protocol: {protocol}\r\n"
        "\r\n"
    )
    sock.sendall(request.encode("ascii"))

    raw = read_http_headers(sock)
    start, headers = parse_http_headers(raw)
    if not start.startswith("HTTP/1.1 101") and not start.startswith("HTTP/1.0 101"):
        raise ValueError(f"websocket upgrade failed: {start}")

    expected = websocket_accept(key)
    actual = headers.get("sec-websocket-accept")
    if actual != expected:
        raise ValueError("bad Sec-WebSocket-Accept")


def run_server(args: argparse.Namespace) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        print(f"listening on ws://{args.host}:{args.port}{args.path}")

        while True:
            conn, addr = listener.accept()
            with conn:
                conn.settimeout(args.timeout)
                print(f"accepted {addr[0]}:{addr[1]}")
                try:
                    server_handshake(conn, args.path, args.protocol)
                    while True:
                        opcode, payload = read_frame(conn)
                        if opcode == OP_TEXT:
                            text = payload.decode("utf-8", "replace")
                            print(f"text {len(payload)} bytes: {text}")
                            send_frame(conn, OP_TEXT, payload)
                            print("echoed message")
                            if args.close_after_echo:
                                send_frame(conn, OP_CLOSE)
                                break
                        elif opcode == OP_BINARY:
                            print(f"binary {len(payload)} bytes")
                            send_frame(conn, OP_BINARY, payload)
                            if args.close_after_echo:
                                send_frame(conn, OP_CLOSE)
                                break
                        elif opcode == OP_PING:
                            send_frame(conn, OP_PONG, payload)
                        elif opcode == OP_CLOSE:
                            send_frame(conn, OP_CLOSE)
                            break
                        else:
                            print(f"opcode {opcode} {len(payload)} bytes")
                except Exception as exc:  # noqa: BLE001 - simple command-line tool
                    print(f"connection failed: {exc}")

            if not args.forever:
                break


def run_client(args: argparse.Namespace) -> None:
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        client_handshake(sock, args.host, args.port, args.path, args.protocol)
        print(f"connected to ws://{args.host}:{args.port}{args.path}")

        payload = args.message.encode("utf-8")
        send_frame(sock, OP_TEXT, payload, mask=True)
        print(f"sent {len(payload)} bytes: {args.message}")

        while True:
            opcode, data = read_frame(sock)
            if opcode == OP_TEXT:
                print(f"received {len(data)} bytes: {data.decode('utf-8', 'replace')}")
                send_frame(sock, OP_CLOSE, mask=True)
                break
            if opcode == OP_PING:
                send_frame(sock, OP_PONG, data, mask=True)
            elif opcode == OP_CLOSE:
                send_frame(sock, OP_CLOSE, mask=True)
                break
            else:
                print(f"received opcode {opcode} with {len(data)} bytes")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Host websocket tester for RT-Smart samples")
    subparsers = parser.add_subparsers(dest="command", required=True)

    server = subparsers.add_parser("server", help="Run PC websocket server for device ws_client")
    server.add_argument("--host", default="0.0.0.0", help="listen address")
    server.add_argument("--port", type=int, default=DEFAULT_PORT, help="listen port")
    server.add_argument("--path", default=DEFAULT_PATH, help="websocket path")
    server.add_argument("--protocol", default=DEFAULT_PROTOCOL, help="websocket subprotocol")
    server.add_argument("--timeout", type=float, default=30.0, help="socket timeout seconds")
    server.add_argument("--forever", action="store_true", help="accept more than one connection")
    server.add_argument(
        "--keep-open",
        dest="close_after_echo",
        action="store_false",
        help="keep the connection open after echoing a message",
    )
    server.set_defaults(func=run_server, close_after_echo=True)

    client = subparsers.add_parser("client", help="Run PC websocket client for device ws_server")
    client.add_argument("host", help="device address")
    client.add_argument("--port", type=int, default=DEFAULT_PORT, help="device port")
    client.add_argument("--path", default=DEFAULT_PATH, help="websocket path")
    client.add_argument("--protocol", default=DEFAULT_PROTOCOL, help="websocket subprotocol")
    client.add_argument("--message", default="Hello from host PC", help="text message to send")
    client.add_argument("--timeout", type=float, default=30.0, help="socket timeout seconds")
    client.set_defaults(func=run_client)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

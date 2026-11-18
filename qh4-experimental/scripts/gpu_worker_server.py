# SPDX-License-Identifier: Apache-2.0
"""Optional lightweight HTTP GPU worker to run on a physical CUDA machine (Colab/RunPod/local GPU).

Expose its port (e.g. via cloudflared/ngrok/LAN) and set QH4_GPU_WORKER_URL on the Next.js server
to let /api/qh4/python-verify dispatch actual CUDA C++ and Triton tests directly to real GPU hardware.
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_actual_cuda_triton_test import run_suite


class GpuWorkerHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            from qh4_experimental.cuda_runner import probe_cuda_toolchain

            self._send_json(200, {"ok": True, "probe": probe_cuda_toolchain()})
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/run-differential":
            self._send_json(404, {"ok": False, "error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            batch_size = max(1, min(4096, int(body.get("batchSize", 128))))
            seed = int(body.get("seed", 2026))
            sm = int(body.get("sm", 80))
            report = run_suite(batch_size=batch_size, seed=seed, target_sm=sm)
            self._send_json(200, {"ok": True, "report": report})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="QH4 remote physical GPU worker server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), GpuWorkerHandler)
    print(f"QH4 GPU worker listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

"""
Headless Client for ComfyUI API in colab-model-station.
Enables programmatic dispatch and retrieval of image generation graphs.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class ComfyUIClient:
    """Client communicating with headless ComfyUI server over HTTP/WebSocket."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.base_url = f"http://{host}:{port}"
        self.client_id = f"station_{int(time.time())}"

    def is_alive(self) -> bool:
        """Check if ComfyUI service is responsive."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def queue_prompt(self, workflow_prompt: Dict[str, Any]) -> str:
        """Submit a prompt graph to the ComfyUI queue and return prompt_id."""
        url = f"{self.base_url}/prompt"
        payload = json.dumps({"prompt": workflow_prompt, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
            return prompt_id

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Fetch execution history for a given prompt_id."""
        url = f"{self.base_url}/history/{prompt_id}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get(prompt_id)
        except Exception:
            return None

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download output image bytes from ComfyUI view endpoint."""
        query = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        url = f"{self.base_url}/view?{query}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read()

    def wait_for_completion(self, prompt_id: str, timeout: int = 300, poll_interval: float = 1.0) -> List[bytes]:
        """Poll history until prompt execution finishes, then download generated images."""
        t_start = time.time()
        while time.time() - t_start < timeout:
            history = self.get_history(prompt_id)
            if history and "outputs" in history:
                images_bytes: List[bytes] = []
                outputs = history["outputs"]
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img_meta in node_output["images"]:
                            fname = img_meta["filename"]
                            subfolder = img_meta.get("subfolder", "")
                            ftype = img_meta.get("type", "output")
                            img_data = self.get_image(fname, subfolder, ftype)
                            images_bytes.append(img_data)
                if images_bytes:
                    return images_bytes
            time.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout} seconds.")

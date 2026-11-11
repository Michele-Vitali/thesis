# This script is directly taken from the OpenVLA github repository and modified as needed.

"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
"""

import os.path

# ruff: noqa: E402
import json_numpy

import scripts.expert.settings as settings

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

# === Utilities ===
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: str | Path) -> str:
    if "v01" in str(openvla_path):
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: str | Path, adapter_path: str | Path | None = None, attn_implementation: str | None = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)

        base_vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map={"": str(self.device)}
        )

        # Merge the LoRA Adapter at runtime!
        if adapter_path is not None:
            print(f"[Merge] Merging the OpenVLA backbone with the adatper: {adapter_path}")
            self.vla = PeftModel.from_pretrained(base_vla, adapter_path)
            self.vla = self.vla.merge_and_unload()
        else:
            self.vla = base_vla

        self.vla = self.vla.to(self.device)

        self.vla.norm_stats = self._load_norm_stats(adapter_path, self.openvla_path)

        if settings.unnorm_key not in self.vla.norm_stats:
            raise RuntimeError(f"unnorm_key '{settings.unnorm_key}' not in {list(self.vla.norm_stats)}")

        self.adapter_path = adapter_path

        model_name = Path(adapter_path).name if adapter_path else Path(openvla_path).name
        print(f"Loaded server with model: {model_name}")

    def _load_norm_stats(self, adapter_path, base_path) -> dict:
        candidates = []
        if adapter_path is not None:
            adapter_path = Path(adapter_path)
            exp_id = adapter_path.name.rsplit("--", 1)[0]
            candidates += [adapter_path / "dataset_statistics.json",
                           settings.runs_root / exp_id / "dataset_statistics.json"]

        if os.path.isdir(base_path):
            candidates.append(Path(base_path) / "dataset_statistics.json")

        for p in candidates:
            if p.is_file():
                with open(p) as f:
                    stats = json.load(f)

                print(f"[OK] Dataset statistics loaded from: {p}")
                return stats

        raise RuntimeError("No dataset_statistics.json found. Looked in:\n " + "\n ".join(map(str, candidates)))

    def predict_action(self, payload: dict[str, Any]) -> JSONResponse:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            # Parse payload components
            image, instruction = payload["image"], payload["instruction"]
            unnorm_key = payload.get("unnorm_key", None)

            # Run VLA Inference
            prompt = get_openvla_prompt(instruction, self.openvla_path)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, temperature=0, do_sample=False)
            if double_encode:
                return JSONResponse({"action": json_numpy.dumps(action)})
            else:
                return JSONResponse({"action": action})
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.warning("Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\nYou can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )

            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    def run(self, host: str = settings.server_host, port: int = settings.server_port) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(self.health)
        print("=" * 60)
        print("[SERVER DEBUG] Host:", repr(host))
        print("[SERVER DEBUG] Port:", repr(port))
        print("=" * 60)
        uvicorn.run(self.app, host=host, port=port, access_log=False)

    def health(self):
        return {"status": "OK", "model": str(self.openvla_path), "adapter": str(self.adapter_path) if self.adapter_path else None}

@dataclass
class DeployConfig:
    # fmt: off
    openvla_path: Union[str, Path] = settings.openvla_path              # If no model is passed, use the fallback model in the settings file!
    adapter_path: Union[str, Path] = settings.adapters_root

    # Server Configuration
    host: str = settings.server_host                                    # Host IP Address
    port: int = settings.server_port                                    # Host Port

    # fmt: on


def run_server_with_cfg(cfg: DeployConfig) -> None:
    server = OpenVLAServer(openvla_path=cfg.openvla_path, adapter_path=cfg.adapter_path)
    server.run(host=cfg.host, port=cfg.port)

@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    run_server_with_cfg(cfg)


if __name__ == "__main__":
    deploy()
from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock
from typing import Any, Deque, Dict, Optional, Tuple

import psutil
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BYTES_PER_PARAM = {
    "FP16": 2,
    "BF16": 2,
    "Q8": 1,
    "Q4": 0.5,
    "Q3": 0.375,
    "Q2": 0.25,
}

DEFAULT_CONTEXT_LEN = 4096
DEFAULT_MODE = "real"
DEFAULT_LANG = "ru"
MODEL_SPECS_VERSION = 1

# Rate limiting
RATE_LIMIT_PER_MINUTE = 60
_rate_limit_lock = Lock()
_rate_limit_data: Dict[str, Deque[float]] = defaultdict(deque)

# Metrics
_metrics_lock = Lock()
_metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "check_model_total": 0,
    "system_info_total": 0,
    "latency_ms_total": 0.0,
}

# API response cache
CACHE_TTL_SECONDS = 300
_cache_lock = Lock()
_check_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SPECS_PATH = os.path.join(BASE_DIR, "model_specs.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")


def log_event(event: str, **kwargs: Any) -> None:
    payload = {"event": event, **kwargs}
    logger.info(json.dumps(payload, ensure_ascii=False))


def build_http_session() -> requests.Session:
    session = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=2,
            backoff_factor=0.25,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
    return session


HTTP_SESSION = build_http_session()


def is_valid_model_key(key: str) -> bool:
    return bool(re.match(r"^[a-z][a-z0-9-]*-(?:\d+x\d+b|\d+b)$", key))


def load_local_specs() -> Dict[str, Dict[str, int]]:
    if not os.path.exists(MODEL_SPECS_PATH):
        save_local_specs({})
        return {}

    try:
        with open(MODEL_SPECS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log_event("specs_load_failed", error=str(e))
        return {}

    # Backward compatibility: old flat format {"llama-7b": {...}}
    if isinstance(data, dict) and "specs" in data and isinstance(data["specs"], dict):
        return {
            k.lower(): v
            for k, v in data["specs"].items()
            if is_valid_model_key(k.lower()) and isinstance(v, dict) and "hidden_size" in v and "layers" in v
        }

    if isinstance(data, dict):
        cleaned: Dict[str, Dict[str, int]] = {}
        for k, v in data.items():
            key = str(k).lower()
            if not is_valid_model_key(key):
                continue
            if isinstance(v, dict) and "hidden_size" in v and "layers" in v:
                cleaned[key] = {"hidden_size": int(v["hidden_size"]), "layers": int(v["layers"])}
        return cleaned

    return {}


def save_local_specs(specs: Dict[str, Dict[str, int]]) -> None:
    payload = {
        "version": MODEL_SPECS_VERSION,
        "specs": dict(sorted(specs.items(), key=lambda x: x[0])),
    }
    with open(MODEL_SPECS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


MODEL_SPECS = load_local_specs()
save_local_specs(MODEL_SPECS)


class SystemInfo(BaseModel):
    ram_gb: float = Field(..., ge=0.1, le=4096)
    vram_gb: float = Field(..., ge=0, le=4096)
    has_gpu: bool
    gpu_name: str = Field(default="Unknown", min_length=1, max_length=200)
    os: Optional[str] = Field(default="Unknown", max_length=200)
    cpu_cores: Optional[int] = Field(default=1, ge=1, le=512)


class ModelCheckRequest(BaseModel):
    model_name: str = Field(..., min_length=2, max_length=200)
    system_info: SystemInfo
    mode: Optional[str] = Field(default=DEFAULT_MODE)
    context_len: Optional[int] = Field(default=DEFAULT_CONTEXT_LEN, ge=256, le=131072)
    language: Optional[str] = Field(default=DEFAULT_LANG)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: Optional[str]) -> str:
        mode = (value or DEFAULT_MODE).lower()
        if mode not in {"real", "ollama"}:
            raise ValueError("mode must be 'real' or 'ollama'")
        return mode

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: Optional[str]) -> str:
        lang = (value or DEFAULT_LANG).lower()
        if lang not in {"ru", "en"}:
            raise ValueError("language must be 'ru' or 'en'")
        return lang


class ModelCheckResponse(BaseModel):
    model_parsed: Dict[str, Any]
    memory_requirements: Dict[str, float]
    recommendation: str
    possible_quantizations: Dict[str, float]
    context_len: int
    language: str


@dataclass
class ParsedModel:
    family: str
    version: str
    param_count_b: float
    is_moe: bool
    moe_experts: int
    moe_active: int
    specialization: str
    quantization: Optional[str]
    raw_name: str


def normalize_model_key(model: ParsedModel) -> str:
    name = model.raw_name.lower()

    moe_match = re.search(r"(\d+)x(\d+)[bB]", name)
    if moe_match:
        experts = moe_match.group(1)
        size = moe_match.group(2)
        return f"{model.family}-{experts}x{size}b"

    size_match = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    if size_match:
        size = str(int(float(size_match.group(1))))
        return f"{model.family}-{size}b"

    return model.family


@lru_cache(maxsize=256)
def fetch_from_huggingface_cached(model_name: str) -> Optional[Dict[str, int]]:
    try:
        parsed = parse_model_name(model_name)
        if not parsed:
            return None

        normalized = normalize_model_key(parsed)
        search_url = f"https://huggingface.co/api/models?search={normalized}&limit=5"
        resp = HTTP_SESSION.get(search_url, timeout=6)
        if resp.status_code != 200:
            return None

        models = resp.json()
        if not models:
            return None

        for m in models:
            repo_id = m.get("id")
            if not repo_id:
                continue

            config_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
            cfg_resp = HTTP_SESSION.get(config_url, timeout=6)
            if cfg_resp.status_code != 200:
                continue

            data = cfg_resp.json()
            hidden = data.get("hidden_size")
            layers = data.get("num_hidden_layers") or data.get("n_layer")

            if hidden and layers:
                return {"hidden_size": int(hidden), "layers": int(layers)}

    except Exception as e:
        log_event("hf_fetch_error", error=str(e), model_name=model_name)

    return None


def get_arch_params(model: ParsedModel) -> Tuple[int, int]:
    normalized_key = normalize_model_key(model)

    if normalized_key in MODEL_SPECS:
        spec = MODEL_SPECS[normalized_key]
        return spec["hidden_size"], spec["layers"]

    for key, spec in MODEL_SPECS.items():
        if key.startswith(model.family + "-") and key.split("-")[-1] == normalized_key.split("-")[-1]:
            return spec["hidden_size"], spec["layers"]

    hf_spec = fetch_from_huggingface_cached(model.raw_name)
    if hf_spec:
        if is_valid_model_key(normalized_key):
            MODEL_SPECS[normalized_key] = hf_spec
            save_local_specs(MODEL_SPECS)
        return hf_spec["hidden_size"], hf_spec["layers"]

    params = model.param_count_b * 1e9
    hidden_size = int((params * 12) ** 0.5)
    num_layers = max(1, hidden_size // 128)

    # Do not persist unknown/invalid keys to avoid noisy specs catalog.
    if model.family != "unknown" and is_valid_model_key(normalized_key):
        MODEL_SPECS[normalized_key] = {"hidden_size": hidden_size, "layers": num_layers}
        save_local_specs(MODEL_SPECS)

    return hidden_size, num_layers


def parse_model_name(name: str) -> Optional[ParsedModel]:
    original = name.strip()
    name = original.lower().replace(":", "-").replace(" ", "-")

    family_match = re.match(r"^([a-zA-Z][a-zA-Z0-9-]*?)(?=[-\d])", name)
    family = family_match.group(1).lower() if family_match else "unknown"

    if name.startswith("llama"):
        family = "llama"
    elif name.startswith("qwen"):
        family = "qwen"
    elif name.startswith("mixtral"):
        family = "mixtral"
    elif name.startswith("mistral"):
        family = "mistral"
    elif name.startswith("gemma"):
        family = "gemma"
    elif name.startswith("phi"):
        family = "phi"

    version = ""
    ver_match = re.search(r"[.-](\d+(?:\.\d+)?(?:-v\d+(?:\.\d+)?)?)", name)
    if ver_match:
        version = ver_match.group(1).lstrip(".-")

    param_count_b = 0.0
    is_moe = False
    moe_experts = 0
    moe_active = 2

    moe_pattern = re.search(r"(\d+)x(\d+)[bB]", name)
    if moe_pattern:
        is_moe = True
        moe_experts = int(moe_pattern.group(1))
        expert_size = float(moe_pattern.group(2))
        param_count_b = moe_experts * expert_size
    else:
        size_match = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
        if size_match:
            param_count_b = float(size_match.group(1))

    if param_count_b <= 0:
        return None

    specialization = "base"
    spec_match = re.search(r"-(instruct|coder|chat|base|vision|audio|tool)", name, re.IGNORECASE)
    if spec_match:
        specialization = spec_match.group(1).lower()

    quantization: Optional[str] = None
    quant_match = re.search(r"-(fp16|bf16|q8|q4|q3|q2|gptq|gguf|awq|int4|int8)", name, re.IGNORECASE)
    if quant_match:
        q = quant_match.group(1).upper()
        if q in ("FP16", "BF16"):
            quantization = q
        elif q in ("Q8", "INT8"):
            quantization = "Q8"
        elif q in ("Q4", "INT4"):
            quantization = "Q4"
        elif q == "Q3":
            quantization = "Q3"
        elif q == "Q2":
            quantization = "Q2"

    if ":" in original:
        quantization = "Q4"

    return ParsedModel(
        family=family,
        version=version,
        param_count_b=param_count_b,
        is_moe=is_moe,
        moe_experts=moe_experts,
        moe_active=moe_active,
        specialization=specialization,
        quantization=quantization,
        raw_name=name,
    )


def estimate_memory(model: ParsedModel, context_len: int = DEFAULT_CONTEXT_LEN, mode: str = DEFAULT_MODE) -> Tuple[float, float]:
    if model.quantization and model.quantization in BYTES_PER_PARAM:
        bytes_per_param = BYTES_PER_PARAM[model.quantization]
    else:
        bytes_per_param = BYTES_PER_PARAM["FP16"]

    total_params = model.param_count_b * 1e9
    model_weights_gb = (total_params * bytes_per_param) / (1024 ** 3)

    hidden_size, num_layers = get_arch_params(model)

    bytes_per_element = 2
    kv_cache_bytes = 2 * num_layers * context_len * hidden_size * bytes_per_element
    kv_cache_gb = kv_cache_bytes / (1024 ** 3)

    kv_cache_gb *= 0.25
    kv_cache_gb = min(kv_cache_gb, model.param_count_b * 0.8)

    if mode == "ollama":
        return model_weights_gb, 0.0

    return model_weights_gb, kv_cache_gb


def get_possible_quantizations(model: ParsedModel, context_len: int) -> Dict[str, float]:
    _, cache_gb = estimate_memory(model, context_len=context_len, mode="real")
    result: Dict[str, float] = {}
    for quant, bytes_per in BYTES_PER_PARAM.items():
        weights_gb = (model.param_count_b * 1e9 * bytes_per) / (1024 ** 3)
        result[quant] = weights_gb + cache_gb
    return result


def t(lang: str, ru: str, en: str) -> str:
    return ru if lang == "ru" else en


def generate_recommendation(
    model: ParsedModel,
    vram_gb: float,
    ram_gb: float,
    has_gpu: bool,
    gpu_name: str,
    mode: str,
    context_len: int,
    lang: str,
) -> str:
    weights_gb, cache_gb = estimate_memory(model, context_len=context_len, mode=mode)
    total_needed = weights_gb + cache_gb
    lines = []

    lines.append(t(lang, f"Модель: {model.raw_name}", f"Model: {model.raw_name}"))
    lines.append(
        t(
            lang,
            f"Параметры: {model.param_count_b:.1f}B" + (" (MoE)" if model.is_moe else ""),
            f"Parameters: {model.param_count_b:.1f}B" + (" (MoE)" if model.is_moe else ""),
        )
    )
    lines.append(t(lang, f"Специализация: {model.specialization}", f"Specialization: {model.specialization}"))
    q_display = model.quantization if model.quantization else t(lang, "не указано (FP16 baseline)", "not set (FP16 baseline)")
    lines.append(t(lang, f"Квантование: {q_display}", f"Quantization: {q_display}"))

    lines.append("")
    lines.append(t(lang, f"Требования памяти (контекст {context_len} токенов):", f"Memory requirements (context {context_len} tokens):"))
    lines.append(t(lang, f"Веса: {weights_gb:.1f} GB", f"Weights: {weights_gb:.1f} GB"))

    if mode == "ollama":
        lines.append(t(lang, "KV cache: скрыт в режиме ollama", "KV cache: hidden in ollama mode"))
        total_display = weights_gb
    else:
        lines.append(t(lang, f"KV cache: {cache_gb:.1f} GB", f"KV cache: {cache_gb:.1f} GB"))
        total_display = total_needed

    lines.append(t(lang, f"Итого VRAM: {total_display:.1f} GB", f"Total VRAM: {total_display:.1f} GB"))

    lines.append("")
    if not has_gpu:
        lines.append(t(lang, f"Система: без GPU, RAM {ram_gb:.0f} GB", f"System: no GPU, RAM {ram_gb:.0f} GB"))
        if ram_gb >= total_needed + 2:
            lines.append(t(lang, "Можно запускать на CPU (медленно).", "CPU inference is possible (slow)."))
            lines.append(t(lang, "Рекомендация: GGUF Q4_K_M и меньший контекст.", "Recommendation: GGUF Q4_K_M and a smaller context."))
        else:
            lines.append(t(lang, "Недостаточно RAM. Лучше облачный API.", "Not enough RAM. Use a cloud API."))
    else:
        lines.append(t(lang, f"GPU: {gpu_name} (VRAM {vram_gb:.0f} GB), RAM {ram_gb:.0f} GB", f"GPU: {gpu_name} (VRAM {vram_gb:.0f} GB), RAM {ram_gb:.0f} GB"))
        if vram_gb >= total_needed:
            lines.append(t(lang, "Модель полностью поместится в VRAM.", "Model fully fits in VRAM."))
        elif vram_gb >= weights_gb:
            if cache_gb > 0:
                reduced_ctx = max(256, int((vram_gb - weights_gb) / cache_gb * context_len))
            else:
                reduced_ctx = context_len
            lines.append(
                t(
                    lang,
                    f"Веса помещаются, но KV cache не влезает. Рекомендуемый контекст: ~{reduced_ctx}.",
                    f"Weights fit, but KV cache does not. Suggested context: ~{reduced_ctx}.",
                )
            )
        else:
            lines.append(t(lang, "Модель не помещается в VRAM.", "Model does not fit in VRAM."))

    return "\n".join(lines)


# ----------------------------
# System info detection
# ----------------------------
def get_gpu_info_windows() -> Tuple[float, str]:
    try:
        cmd = "wmic path win32_videocontroller get name,adapterram /format:csv"
        output = subprocess.check_output(cmd, shell=True, text=True, encoding="cp866", errors="ignore")
        lines = output.strip().split("\n")
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) >= 3:
                name = parts[1].strip()
                vram_bytes = parts[2].strip()
                if vram_bytes.isdigit():
                    return int(vram_bytes) / (1024 ** 3), name
    except Exception:
        pass
    return 0.0, "Unknown"


def get_gpu_info_macos() -> Tuple[float, str]:
    try:
        output = subprocess.check_output(["system_profiler", "SPDisplaysDataType", "-json"], text=True)
        data = json.loads(output)
        gpus = data.get("SPDisplaysDataType", [])
        if gpus:
            gpu = gpus[0]
            name = gpu.get("sppci_model", "Unknown")
            vram_str = gpu.get("spdisplays_vram", "")
            if not vram_str and ("Apple" in name or "M" in name):
                total_ram = psutil.virtual_memory().total
                return total_ram / (1024 ** 3), name

            match = re.search(r"(\d+(?:\.\d+)?)\s*(GB|MB)", vram_str, re.IGNORECASE)
            if match:
                value, unit = match.groups()
                mem_gb = float(value) if unit.upper() == "GB" else float(value) / 1024
                return mem_gb, name
            return 0.0, name
    except Exception:
        pass
    return 0.0, "Unknown"


def get_gpu_info_linux() -> Tuple[float, str]:
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        output = subprocess.check_output(cmd, shell=True, text=True)
        line = output.strip().split("\n")[0]
        name, mem = line.split(",")
        return float(mem.strip().split()[0]) / 1024, name.strip()
    except Exception:
        pass

    try:
        cmd = "lspci -v | grep -A 10 'VGA compatible controller'"
        output = subprocess.check_output(cmd, shell=True, text=True)
        match = re.search(r"(\d+)MB", output)
        vram_gb = int(match.group(1)) / 1024 if match else 0.0
        name_match = re.search(r"VGA compatible controller: (.+?)(?:\n|$)", output)
        name = name_match.group(1) if name_match else "Unknown"
        return vram_gb, name
    except Exception:
        pass

    return 0.0, "Unknown"


def get_system_info() -> Dict[str, Any]:
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    os_name = platform.system() + " " + platform.release()
    cpu_cores = psutil.cpu_count(logical=True) or 1

    system = platform.system()
    if system == "Windows":
        vram_gb, gpu_name = get_gpu_info_windows()
    elif system == "Darwin":
        vram_gb, gpu_name = get_gpu_info_macos()
    else:
        vram_gb, gpu_name = get_gpu_info_linux()

    has_gpu = gpu_name.lower() != "unknown"
    if has_gpu and vram_gb == 0:
        vram_gb = 1.0

    return {
        "ram_gb": round(ram_gb, 2),
        "vram_gb": round(vram_gb, 2),
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "os": os_name,
        "cpu_cores": int(cpu_cores),
    }


def get_client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if xff:
        return xff
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(ip: str) -> bool:
    now = time.time()
    with _rate_limit_lock:
        bucket = _rate_limit_data[ip]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
    return True


def make_cache_key(req: ModelCheckRequest) -> str:
    return json.dumps(req.model_dump(), sort_keys=True, ensure_ascii=False)


def get_cached_response(key: str) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _cache_lock:
        hit = _check_cache.get(key)
        if not hit:
            return None
        ts, value = hit
        if now - ts > CACHE_TTL_SECONDS:
            _check_cache.pop(key, None)
            return None
        return value


def put_cached_response(key: str, value: Dict[str, Any]) -> None:
    now = time.time()
    with _cache_lock:
        _check_cache[key] = (now, value)


app = FastAPI(title="LLM Model Checker")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def telemetry_middleware(request: Request, call_next):
    start = time.perf_counter()
    ip = get_client_ip(request)

    if request.url.path.startswith("/api/") and not enforce_rate_limit(ip):
        with _metrics_lock:
            _metrics["requests_total"] += 1
            _metrics["errors_total"] += 1
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again in a minute."})

    try:
        response = await call_next(request)
    except Exception:
        with _metrics_lock:
            _metrics["requests_total"] += 1
            _metrics["errors_total"] += 1
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    with _metrics_lock:
        _metrics["requests_total"] += 1
        _metrics["latency_ms_total"] += latency_ms
        if response.status_code >= 400:
            _metrics["errors_total"] += 1

    log_event(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency_ms, 2),
        ip=ip,
    )
    return response


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/system_info")
async def api_system_info():
    with _metrics_lock:
        _metrics["system_info_total"] += 1
    return get_system_info()


@app.get("/api/metrics")
async def api_metrics():
    with _metrics_lock:
        total = _metrics["requests_total"]
        avg_latency_ms = (_metrics["latency_ms_total"] / total) if total else 0.0
        return {
            **_metrics,
            "avg_latency_ms": round(avg_latency_ms, 2),
            "cache_items": len(_check_cache),
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
        }


@app.post("/api/check_model", response_model=ModelCheckResponse)
async def check_model(req: ModelCheckRequest):
    with _metrics_lock:
        _metrics["check_model_total"] += 1

    model = parse_model_name(req.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Could not parse model name")

    cache_key = make_cache_key(req)
    cached = get_cached_response(cache_key)
    if cached:
        return cached

    mode = req.mode or DEFAULT_MODE
    context_len = req.context_len or DEFAULT_CONTEXT_LEN
    lang = req.language or DEFAULT_LANG

    weights, cache = estimate_memory(model, context_len=context_len, mode=mode)
    total = weights + cache
    recommendation = generate_recommendation(
        model=model,
        vram_gb=req.system_info.vram_gb,
        ram_gb=req.system_info.ram_gb,
        has_gpu=req.system_info.has_gpu,
        gpu_name=req.system_info.gpu_name,
        mode=mode,
        context_len=context_len,
        lang=lang,
    )
    possible = get_possible_quantizations(model, context_len=context_len)

    response = {
        "model_parsed": {
            "family": model.family,
            "version": model.version,
            "param_count_b": model.param_count_b,
            "is_moe": model.is_moe,
            "specialization": model.specialization,
            "quantization": model.quantization,
        },
        "memory_requirements": {
            "weights_gb": weights,
            "kv_cache_gb": cache,
            "total_vram_gb": total,
        },
        "recommendation": recommendation,
        "possible_quantizations": possible,
        "context_len": context_len,
        "language": lang,
    }

    put_cached_response(cache_key, response)
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

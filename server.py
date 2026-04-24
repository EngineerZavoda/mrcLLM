from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import json
import os
import requests
from functools import lru_cache

# ----------------------------
# Конфигурация
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BYTES_PER_PARAM = {
    "FP16": 2,
    "BF16": 2,
    "Q8": 1,
    "Q4": 0.5,
    "Q3": 0.375,
    "Q2": 0.25
}

DEFAULT_QUANT = None  # квантование не задаётся по умолчанию
MEMORY_MODE = "real"  # "real" или "ollama"
KV_CACHE_BASE_GB = 1.5   # для 7B, контекст 4096
BASE_PARAMS = 7e9

BASE_CONTEXT = 4096

# ----------------------------
# Архитектура моделей (загрузка из JSON + автообновление)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SPECS_PATH = os.path.join(BASE_DIR, "model_specs.json")
MODEL_SPECS_URL = "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/llama/configuration_llama.py"


def load_local_specs() -> Dict[str, Dict[str, int]]:
    # если файла нет — создаём пустой JSON
    if not os.path.exists(MODEL_SPECS_PATH):
        try:
            with open(MODEL_SPECS_PATH, "w") as f:
                json.dump({}, f, indent=2)
            print(f"🆕 Created empty specs file at {MODEL_SPECS_PATH}")
        except Exception as e:
            print(f"❌ Failed to create specs file: {e}")
            return {}

    # читаем файл
    try:
        with open(MODEL_SPECS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load specs: {e}")
        return {}


def save_local_specs(specs: Dict[str, Dict[str, int]]):
    try:
        with open(MODEL_SPECS_PATH, "w") as f:
            json.dump(specs, f, indent=2)
        print(f"💾 Saved specs to {MODEL_SPECS_PATH}")
    except Exception as e:
        print(f"❌ Failed to save specs: {e}")


def fetch_remote_specs() -> Dict[str, Dict[str, int]]:
    """
    Загружает спеки моделей из удалённого JSON (GitHub fallback).
    Если не удалось — возвращает пустой словарь.
    """
    try:
        url = "https://raw.githubusercontent.com/pvs-dev/llm-model-specs/main/specs.json"
        resp = requests.get(url, timeout=5)

        if resp.status_code != 200:
            return {}

        data = resp.json()

        # ожидаемый формат:
        # {
        #   "llama-7b": {"hidden_size": 4096, "layers": 32},
        #   ...
        # }
        valid = {}

        for k, v in data.items():
            if (
                isinstance(v, dict)
                and "hidden_size" in v
                and "layers" in v
            ):
                valid[k.lower()] = {
                    "hidden_size": int(v["hidden_size"]),
                    "layers": int(v["layers"]),
                }

        return valid

    except Exception:
        return {}

@lru_cache(maxsize=128)
def fetch_from_huggingface_cached(model_name: str) -> Optional[Dict[str, int]]:
    """
    Умный поиск модели в HuggingFace через API + извлечение config.json
    Работает для любых моделей без ручного mapping
    """
    try:
        parsed = parse_model_name(model_name)
        if not parsed:
            print("❌ HF: parse failed")
            return None

        normalized = normalize_model_key(parsed)
        print(f"🔍 HF search for: {normalized}")

        # --- 1. поиск моделей через HF API ---
        search_url = f"https://huggingface.co/api/models?search={normalized}&limit=5"
        resp = requests.get(search_url, timeout=5)

        if resp.status_code != 200:
            print("❌ HF search failed")
            return None

        models = resp.json()
        if not models:
            print("❌ HF: no models found")
            return None

        # --- 2. перебираем кандидатов ---
        for m in models:
            repo_id = m.get("id")
            if not repo_id:
                continue

            config_url = f"https://huggingface.co/{repo_id}/raw/main/config.json"

            try:
                cfg_resp = requests.get(config_url, timeout=5)
                if cfg_resp.status_code != 200:
                    continue

                data = cfg_resp.json()

                # --- 3. извлекаем архитектуру ---
                hidden = data.get("hidden_size")
                layers = data.get("num_hidden_layers") or data.get("n_layer")

                if hidden and layers:
                    print(f"✅ HF match: {repo_id}")
                    return {
                        "hidden_size": int(hidden),
                        "layers": int(layers),
                    }

            except Exception:
                continue

        print("⚠️ HF: no valid config found")
        return None

    except Exception as e:
        print(f"❌ HF resolver error: {e}")
        return None

def init_model_specs() -> Dict[str, Dict[str, int]]:
    local_specs = load_local_specs()
    remote_specs = fetch_remote_specs()

    merged = local_specs.copy()

    if remote_specs:
        merged.update(remote_specs)

    # гарантированно сохраняем файл
    save_local_specs(merged)

    return merged


MODEL_SPECS = init_model_specs()

def normalize_model_key(model: "ParsedModel") -> str:
    """
    Приводит название модели к нормализованному виду.
    Пример:
    llama-3.3-70b-instruct-q4_k_m -> llama-70b
    mixtral-8x7b-instruct -> mixtral-8x7b
    """
    name = model.raw_name.lower()

    # MoE (например 8x7b)
    moe_match = re.search(r"(\d+)x(\d+)[bB]", name)
    if moe_match:
        experts = moe_match.group(1)
        size = moe_match.group(2)
        return f"{model.family}-{experts}x{size}b"

    # обычные модели (7b, 13b, 70b)
    size_match = re.search(r"(\d+(?:\.\d+)?)[bB]", name)
    if size_match:
        size = size_match.group(1)
        # убираем .0 если есть
        size = str(int(float(size)))
        return f"{model.family}-{size}b"

    return model.family

def get_arch_params(model: "ParsedModel") -> Tuple[int, int]:
    """Возвращает hidden_size и num_layers (сначала из базы, иначе эвристика)"""
    name = model.raw_name.lower()

    normalized_key = normalize_model_key(model)

    # точное совпадение
    if normalized_key in MODEL_SPECS:
        spec = MODEL_SPECS[normalized_key]
        return spec["hidden_size"], spec["layers"]

    # fallback: совпадение по семейству + размеру (без ложных совпадений)
    for key, spec in MODEL_SPECS.items():
        if key.startswith(model.family) and key.split("-")[-1] == normalized_key.split("-")[-1]:
            return spec["hidden_size"], spec["layers"]

    # --- попытка получить из HuggingFace ---
    hf_spec = fetch_from_huggingface_cached(model.raw_name)
    if hf_spec:
        # сохраняем в кэш
        normalized_key = normalize_model_key(model)
        MODEL_SPECS[normalized_key] = hf_spec
        save_local_specs(MODEL_SPECS)
        print(f"🧠 Auto-added spec for {normalized_key} from HuggingFace")
        return hf_spec["hidden_size"], hf_spec["layers"]

    # --- fallback (эвристика + автосохранение) ---
    params = model.param_count_b * 1e9
    hidden_size = int((params * 12) ** 0.5)
    num_layers = max(1, hidden_size // 128)

    normalized_key = normalize_model_key(model)

    # сохраняем даже эвристику — чтобы JSON всегда заполнялся
    MODEL_SPECS[normalized_key] = {
        "hidden_size": hidden_size,
        "layers": num_layers
    }
    save_local_specs(MODEL_SPECS)
    print(f"⚠️ Auto-added heuristic spec for {normalized_key}")

    return hidden_size, num_layers

# ----------------------------
# Модели данных для API
# ----------------------------
class SystemInfo(BaseModel):
    ram_gb: float
    vram_gb: float
    has_gpu: bool
    gpu_name: str
    os: Optional[str] = None
    cpu_cores: Optional[int] = None

class ModelCheckRequest(BaseModel):
    model_name: str
    system_info: SystemInfo
    mode: Optional[str] = None

class ModelCheckResponse(BaseModel):
    model_parsed: Dict[str, Any]
    memory_requirements: Dict[str, float]
    recommendation: str
    possible_quantizations: Dict[str, float]

# ----------------------------
# Парсинг названия модели
# ----------------------------
@dataclass
class ParsedModel:
    family: str
    version: str
    param_count_b: float
    is_moe: bool
    moe_experts: int
    moe_active: int
    specialization: str
    quantization: str
    raw_name: str

def parse_model_name(name: str) -> Optional[ParsedModel]:
    name = name.strip()
    # нормализация ввода (Ollama / user-friendly)
    name = name.lower()
    name = name.replace(":", "-")
    name = name.replace(" ", "-")
    raw = name

    # Семейство
    family_match = re.match(r'^([a-zA-Z][a-zA-Z0-9-]*?)(?=[-\d])', name)
    family = family_match.group(1) if family_match else "Unknown"

    # нормализация семейства (под Ollama / HF / GGUF)
    fname = name.lower()
    if fname.startswith("llama"):
        family = "llama"
    elif fname.startswith("qwen"):
        family = "qwen"
    elif fname.startswith("mixtral"):
        family = "mixtral"
    elif fname.startswith("mistral"):
        family = "mistral"
    elif fname.startswith("gemma"):
        family = "gemma"
    elif fname.startswith("phi"):
        family = "phi"

    # Версия
    version = ""
    ver_match = re.search(r'[.-](\d+(?:\.\d+)?(?:-v\d+(?:\.\d+)?)?)', name)
    if ver_match:
        version = ver_match.group(1).lstrip('.-')

    # Размер и MoE
    param_count_b = 0.0
    is_moe = False
    moe_experts = 0
    moe_active = 2

    moe_pattern = re.search(r'(\d+)x(\d+)[bB]', name)
    if moe_pattern:
        is_moe = True
        moe_experts = int(moe_pattern.group(1))
        expert_size = float(moe_pattern.group(2))
        param_count_b = moe_experts * expert_size
    else:
        size_match = re.search(r'(\d+(?:\.\d+)?)[bB]', name)
        if size_match:
            param_count_b = float(size_match.group(1))

    if param_count_b == 0:
        return None

    # Специализация
    specialization = "base"
    spec_match = re.search(r'-(instruct|coder|chat|base|vision|audio|tool)', name, re.IGNORECASE)
    if spec_match:
        specialization = spec_match.group(1).lower()

    # Квантование
    quantization = None
    quant_match = re.search(r'-(fp16|bf16|q8|q4|q3|q2|gptq|gguf|awq|int4|int8)', name, re.IGNORECASE)
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
        else:
            quantization = None

    # Ollama-style формат (например qwen3.5:9b)
    if ":" in name:
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
        raw_name=raw
    )

# ----------------------------
# Расчёт памяти
# ----------------------------
def estimate_memory(model: ParsedModel, context_len: int = 4096, mode: str = MEMORY_MODE) -> Tuple[float, float]:
    # --- 1. байты на параметр ---
    if model.quantization and model.quantization in BYTES_PER_PARAM:
        bytes_per_param = BYTES_PER_PARAM[model.quantization]
    else:
        # если квантование не указано — считаем FP16 как baseline
        bytes_per_param = BYTES_PER_PARAM["FP16"]

    # --- 2. веса ---
    total_params = model.param_count_b * 1e9
    model_weights_gb = (total_params * bytes_per_param) / (1024**3)

    # --- 3. архитектура модели ---
    hidden_size, num_layers = get_arch_params(model)

    # --- 4. KV cache (точная формула) ---
    bytes_per_element = 2  # обычно FP16

    kv_cache_bytes = (
        2 * num_layers * context_len * hidden_size * bytes_per_element
    )

    kv_cache_gb = kv_cache_bytes / (1024**3)

    # более реалистичная модель (как в Ollama / llama.cpp)
    # KV cache обычно значительно меньше из-за оптимизаций
    kv_cache_gb *= 0.25  # эмпирический коэффициент

    # ограничение (реалистичное)
    kv_cache_gb = min(kv_cache_gb, model.param_count_b * 0.8)

    # режим Ollama: считаем только веса
    if mode == "ollama":
        return model_weights_gb, 0.0

    return model_weights_gb, kv_cache_gb

def get_possible_quantizations(model: ParsedModel) -> Dict[str, float]:
    """Для заданной модели вычисляет необходимую VRAM для разных типов квантования"""
    result = {}
    for quant, bytes_per in BYTES_PER_PARAM.items():
        weights_gb = (model.param_count_b * 1e9 * bytes_per) / (1024**3)
        # используем уже более реалистичный cache (не зависит от квантования)
        _, cache_gb = estimate_memory(model, mode="real")
        result[quant] = weights_gb + cache_gb
    return result

def generate_recommendation(
    model: ParsedModel,
    vram_gb: float,
    ram_gb: float,
    has_gpu: bool,
    gpu_name: str,
    mode: str = MEMORY_MODE
) -> str:
    weights_gb, cache_gb = estimate_memory(model, mode=mode)
    total_needed = weights_gb + cache_gb
    lines = []

    lines.append(f"📊 Модель: {model.raw_name}")
    lines.append(f"   Параметров: {model.param_count_b:.1f}B" + (" (MoE)" if model.is_moe else ""))
    lines.append(f"   Специализация: {model.specialization}")
    q_display = model.quantization if model.quantization else "не указано (FP16 baseline)"
    lines.append(f"   Квантование: {q_display}")

    lines.append(f"\n💾 Требования к памяти (контекст 4096 токенов):")
    lines.append(f"   Веса: {weights_gb:.1f} GB")
    if mode == "ollama":
        lines.append("   KV cache: (скрыт в режиме Ollama)")
        lines.append(f"   Итого VRAM: {weights_gb:.1f} GB")
    else:
        lines.append(f"   KV cache: {cache_gb:.1f} GB")
        lines.append(f"   Итого VRAM: {total_needed:.1f} GB")
        lines.append("   ℹ️ Ollama обычно показывает только вес модели (без KV cache)")

    if not has_gpu:
        lines.append(f"\n🖥️ Система: нет дискретного GPU, ОЗУ {ram_gb:.0f} GB")
        if ram_gb >= total_needed + 2:
            lines.append("✅ Можно запустить на CPU (llama.cpp). Скорость низкая (~2-5 токен/с).")
            lines.append("💡 Используйте GGUF Q4_K_M и уменьшите контекст.")
        else:
            lines.append("❌ Недостаточно ОЗУ. Рекомендуется облачный API.")
    else:
        lines.append(f"\n🖥️ GPU: {gpu_name} (VRAM {vram_gb:.0f} GB), ОЗУ {ram_gb:.0f} GB")
        if vram_gb >= total_needed:
            lines.append("✅ Модель полностью помещается в VRAM. Отличная скорость.")
        elif vram_gb >= weights_gb:
            lines.append(f"⚠️ Веса влезают, но KV cache требует {cache_gb:.1f} GB. Уменьшите контекст до ~{int((vram_gb - weights_gb) / cache_gb * 4096)} токенов.")
        else:
            lines.append("❌ Модель не влезает в VRAM.")
            # Показать альтернативные квантования
            quants = get_possible_quantizations(model)
            better = [(q, needed) for q, needed in quants.items() if needed <= vram_gb and q != model.quantization]
            if better:
                lines.append("🔧 Попробуйте квантование: " + ", ".join([f"{q} ({needed:.1f} GB)" for q, needed in better[:2]]))
            else:
                lines.append("💡 Даже Q2 не влезает. Используйте облачный API или CPU (если ОЗУ ≥ 32 GB).")
    return "\n".join(lines)

# ----------------------------
# FastAPI приложение
# ----------------------------
app = FastAPI(title="LLM Model Checker")

@app.post("/api/check_model", response_model=ModelCheckResponse)
async def check_model(req: ModelCheckRequest):
    logger.info(f"Checking model: {req.model_name}")
    model = parse_model_name(req.model_name)
    if not model:
        raise HTTPException(status_code=400, detail="Could not parse model name")

    mode = req.mode if req.mode else MEMORY_MODE
    weights, cache = estimate_memory(model, mode=mode)
    total = weights + cache
    recommendation = generate_recommendation(
        model,
        req.system_info.vram_gb,
        req.system_info.ram_gb,
        req.system_info.has_gpu,
        req.system_info.gpu_name,
        mode=mode
    )
    possible = get_possible_quantizations(model)

    return ModelCheckResponse(
        model_parsed={
            "family": model.family,
            "version": model.version,
            "param_count_b": model.param_count_b,
            "is_moe": model.is_moe,
            "specialization": model.specialization,
            "quantization": model.quantization
        },
        memory_requirements={
            "weights_gb": weights,
            "kv_cache_gb": cache,
            "total_vram_gb": total
        },
        recommendation=recommendation,
        possible_quantizations=possible
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
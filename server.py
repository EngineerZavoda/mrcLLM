import re
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
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

DEFAULT_QUANT = "FP16"
KV_CACHE_BASE_GB = 1.5   # для 7B, контекст 4096
BASE_PARAMS = 7e9
BASE_CONTEXT = 4096

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
    raw = name

    # Семейство
    family_match = re.match(r'^([a-zA-Z][a-zA-Z0-9-]*?)(?=[-\d])', name)
    family = family_match.group(1) if family_match else "Unknown"

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
    quantization = DEFAULT_QUANT
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
            quantization = "Q4"  # GPTQ/GGUF/AWQ считаем за Q4
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
def estimate_memory(model: ParsedModel, context_len: int = 4096) -> Tuple[float, float]:
    bytes_per_param = BYTES_PER_PARAM.get(model.quantization, BYTES_PER_PARAM[DEFAULT_QUANT])
    total_params = model.param_count_b * 1e9
    model_weights_gb = (total_params * bytes_per_param) / (1024**3)

    # KV cache: масштабируем от базового
    kv_cache_gb = KV_CACHE_BASE_GB * (total_params / BASE_PARAMS) * (context_len / BASE_CONTEXT)
    kv_cache_gb = min(kv_cache_gb, 32.0)
    return model_weights_gb, kv_cache_gb

def get_possible_quantizations(model: ParsedModel) -> Dict[str, float]:
    """Для заданной модели вычисляет необходимую VRAM для разных типов квантования"""
    result = {}
    for quant, bytes_per in BYTES_PER_PARAM.items():
        weights_gb = (model.param_count_b * 1e9 * bytes_per) / (1024**3)
        _, cache_gb = estimate_memory(model)  # cache не зависит от квантования весов
        result[quant] = weights_gb + cache_gb
    return result

def generate_recommendation(model: ParsedModel, vram_gb: float, ram_gb: float, has_gpu: bool, gpu_name: str) -> str:
    weights_gb, cache_gb = estimate_memory(model)
    total_needed = weights_gb + cache_gb
    lines = []

    lines.append(f"📊 Модель: {model.raw_name}")
    lines.append(f"   Параметров: {model.param_count_b:.1f}B" + (" (MoE)" if model.is_moe else ""))
    lines.append(f"   Специализация: {model.specialization}")
    lines.append(f"   Квантование: {model.quantization}")

    lines.append(f"\n💾 Требования к памяти (контекст 4096 токенов):")
    lines.append(f"   Веса: {weights_gb:.1f} GB")
    lines.append(f"   KV cache: {cache_gb:.1f} GB")
    lines.append(f"   Итого VRAM: {total_needed:.1f} GB")

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
    
    weights, cache = estimate_memory(model)
    total = weights + cache
    recommendation = generate_recommendation(
        model,
        req.system_info.vram_gb,
        req.system_info.ram_gb,
        req.system_info.has_gpu,
        req.system_info.gpu_name
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
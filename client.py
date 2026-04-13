#!/usr/bin/env python3
"""
LLM Model Checker Client - кроссплатформенный агент для сбора характеристик ПК
Поддерживает macOS (включая Apple Silicon), Windows, Linux.
"""

import sys
import platform
import subprocess
import re
import json
import requests

try:
    import psutil
except ImportError:
    print("❌ Ошибка: требуется установить psutil. Выполните: pip install psutil")
    sys.exit(1)

# ----------------------------------------------------------------------
# Функции определения GPU на разных ОС
# ----------------------------------------------------------------------

def get_gpu_info_windows():
    """Windows: через wmic / PowerShell"""
    try:
        cmd = "wmic path win32_videocontroller get name,adapterram /format:csv"
        output = subprocess.check_output(cmd, shell=True, text=True, encoding='cp866', errors='ignore')
        lines = output.strip().split('\n')
        for line in lines[1:]:
            parts = line.split(',')
            if len(parts) >= 3:
                name = parts[1].strip()
                vram_bytes = parts[2].strip()
                if vram_bytes.isdigit():
                    vram_gb = int(vram_bytes) / (1024**3)
                    return vram_gb, name
        # PowerShell fallback
        cmd = "powershell \"Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json\""
        output = subprocess.check_output(cmd, shell=True, text=True)
        data = json.loads(output)
        if isinstance(data, list):
            for gpu in data:
                name = gpu.get('Name', 'Unknown')
                vram_bytes = gpu.get('AdapterRAM', 0)
                if vram_bytes:
                    vram_gb = vram_bytes / (1024**3)
                    return vram_gb, name
    except Exception:
        pass
    return 0.0, "Unknown"

def get_gpu_info_macos():
    """macOS: через system_profiler SPDisplaysDataType, включая Apple Silicon"""
    try:
        cmd = ["system_profiler", "SPDisplaysDataType", "-json"]
        output = subprocess.check_output(cmd, text=True)
        data = json.loads(output)
        gpus = data.get("SPDisplaysDataType", [])
        if gpus:
            gpu = gpus[0]
            name = gpu.get("sppci_model", "Unknown")
            vram_str = gpu.get("spdisplays_vram", "")
            
            # Apple Silicon: нет отдельной VRAM, используем общую ОЗУ
            if not vram_str and ("Apple" in name or "M" in name):
                total_ram = psutil.virtual_memory().total
                vram_gb = total_ram / (1024**3)
                return vram_gb, name
            
            # Обычные GPU (Intel, AMD)
            match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|MB)', vram_str, re.IGNORECASE)
            if match:
                value, unit = match.groups()
                vram_gb = float(value) if unit.upper() == "GB" else float(value) / 1024
            else:
                vram_gb = 0.0
            return vram_gb, name
    except Exception as e:
        pass
    return 0.0, "Unknown"

def get_gpu_info_linux():
    """Linux: через nvidia-smi или lspci"""
    # Попробуем nvidia-smi
    try:
        cmd = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
        output = subprocess.check_output(cmd, shell=True, text=True)
        line = output.strip().split('\n')[0]
        name, mem = line.split(',')
        mem_gb = float(mem.strip().split()[0]) / 1024  # из MiB в GB
        return mem_gb, name.strip()
    except Exception:
        pass
    # fallback: lspci
    try:
        cmd = "lspci -v | grep -A 10 'VGA compatible controller'"
        output = subprocess.check_output(cmd, shell=True, text=True)
        match = re.search(r'(\d+)MB', output)
        vram_gb = int(match.group(1)) / 1024 if match else 0.0
        name_match = re.search(r'VGA compatible controller: (.+?)(?:\n|$)', output)
        name = name_match.group(1) if name_match else "Unknown"
        return vram_gb, name
    except Exception:
        pass
    return 0.0, "Unknown"

def get_gpu_info():
    """Кроссплатформенный сбор информации о GPU"""
    system = platform.system()
    if system == "Windows":
        vram_gb, gpu_name = get_gpu_info_windows()
    elif system == "Darwin":  # macOS
        vram_gb, gpu_name = get_gpu_info_macos()
    else:  # Linux и другие
        vram_gb, gpu_name = get_gpu_info_linux()
    
    has_gpu = gpu_name.lower() != "unknown"
    # Если vram_gb == 0, но GPU есть (например, старый Intel без отдельной памяти), считаем 1GB для оценки
    if has_gpu and vram_gb == 0:
        vram_gb = 1.0
    return vram_gb, has_gpu, gpu_name

def get_system_info():
    ram_gb = psutil.virtual_memory().total / (1024**3)
    vram_gb, has_gpu, gpu_name = get_gpu_info()
    os_name = platform.system() + " " + platform.release()
    cpu_cores = psutil.cpu_count(logical=True)
    return {
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "has_gpu": has_gpu,
        "gpu_name": gpu_name,
        "os": os_name,
        "cpu_cores": cpu_cores
    }

def main():
    if len(sys.argv) < 2:
        print("Использование: python client.py <название_модели> [--server URL]")
        print("Пример: python client.py llama-3.3-70b-instruct-Q4_K_M")
        sys.exit(1)
    
    model_name = sys.argv[1]
    server_url = "http://localhost:8000"
    if "--server" in sys.argv:
        idx = sys.argv.index("--server")
        if idx+1 < len(sys.argv):
            server_url = sys.argv[idx+1]
    
    print("🔍 Сбор характеристик системы...")
    sys_info = get_system_info()
    print(f"   Операционная система: {sys_info['os']}")
    print(f"   ОЗУ: {sys_info['ram_gb']:.1f} GB")
    print(f"   CPU ядер: {sys_info['cpu_cores']}")
    if sys_info['has_gpu']:
        print(f"   GPU: {sys_info['gpu_name']} (доступно VRAM: {sys_info['vram_gb']:.1f} GB)")
    else:
        print("   GPU: не обнаружен (будет использоваться CPU)")
    
    payload = {
        "model_name": model_name,
        "system_info": sys_info
    }
    
    print(f"📡 Отправка запроса на {server_url}/api/check_model ...")
    try:
        response = requests.post(f"{server_url}/api/check_model", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТ:")
        print(data["recommendation"])
        if "possible_quantizations" in data:
            print("\n📊 Альтернативные варианты квантования (требуемая VRAM):")
            for quant, needed in sorted(data["possible_quantizations"].items(), key=lambda x: x[1]):
                print(f"   {quant}: {needed:.1f} GB")
        print("="*60)
    except requests.exceptions.ConnectionError:
        print(f"❌ Не удалось подключиться к серверу. Убедитесь, что сервер запущен (python server.py) и доступен по адресу {server_url}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
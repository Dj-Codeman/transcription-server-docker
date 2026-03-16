.PHONY: help venv dirs setup install-base install-nvidia install-amd install-all check check-amd-env reset-venv clean

PYTHON_BIN ?= python3.12
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
ROCM_INDEX ?= https://download.pytorch.org/whl/rocm6.2

help:
	@printf "Targets:\n"
	@printf "  make setup           Create venv, upgrade pip, and create runtime dirs\n"
	@printf "  make install-base    Install base deps (CPU path)\n"
	@printf "  make install-nvidia  Install faster-whisper for NVIDIA/CPU\n"
	@printf "  make install-amd     Install ROCm PyTorch + openai-whisper\n"
	@printf "  make install-all     Install both NVIDIA and AMD backends\n"
	@printf "  make check-amd-env   Validate Python/platform for ROCm wheels\n"
	@printf "  make check           Compile-check transcribe.py\n"
	@printf "  make reset-venv      Recreate venv with $(PYTHON_BIN)\n"
	@printf "  make clean           Remove generated cache files\n"
	@printf "\n"
	@printf "Configurable vars:\n"
	@printf "  PYTHON_BIN=%s\n" "$(PYTHON_BIN)"
	@printf "  ROCM_INDEX=%s\n" "$(ROCM_INDEX)"
	@printf "\n"
	@printf "Activate venv in your shell:\n"
	@printf "  source $(VENV)/bin/activate\n"

venv:
	@test -d "$(VENV)" || $(PYTHON_BIN) -m venv "$(VENV)"
	@$(PYTHON) -c 'import sys; req=(3,12); cur=sys.version_info[:2];\
	assert cur==req, f"Expected Python {req[0]}.{req[1]} in $(VENV), found {cur[0]}.{cur[1]}. Run make reset-venv PYTHON_BIN=$(PYTHON_BIN)"'
	@$(PIP) install --upgrade pip setuptools wheel

dirs:
	@mkdir -p /tmp/transcriptions

setup: venv dirs

install-base: setup
	@$(PIP) install faster-whisper

install-nvidia: setup
	@$(PIP) install faster-whisper

install-amd: setup
	@$(MAKE) check-amd-env
	@$(PIP) install --index-url $(ROCM_INDEX) torch torchvision torchaudio
	@$(PIP) install openai-whisper

install-all: setup
	@$(PIP) install faster-whisper
	@$(MAKE) check-amd-env
	@$(PIP) install --index-url $(ROCM_INDEX) torch torchvision torchaudio
	@$(PIP) install openai-whisper

check-amd-env: setup
	@$(PYTHON) -c 'import platform,sys;\
	maj,min=sys.version_info[:2];\
	assert (maj,min) in {(3,10),(3,11),(3,12)}, f"ROCm PyTorch wheels usually support Python 3.10-3.12; found {maj}.{min}";\
	assert sys.platform.startswith("linux"), f"ROCm wheels require Linux; found {sys.platform}";\
	assert platform.machine()=="x86_64", f"ROCm wheels generally require x86_64; found {platform.machine()}";\
	print("ROCm environment looks compatible")'

check: setup
	@$(PYTHON) -m py_compile transcribe.py

reset-venv:
	@rm -rf "$(VENV)"
	@$(MAKE) setup PYTHON_BIN=$(PYTHON_BIN)

clean:
	@rm -rf __pycache__

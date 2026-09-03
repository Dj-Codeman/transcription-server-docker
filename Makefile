.PHONY: help venv dirs setup install-base install-nvidia install-amd install-all check check-amd-env reset-venv clean docker-build docker-run install-client uninstall-client

PYTHON_BIN ?= python3.14
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
ROCM_INDEX ?= https://download.pytorch.org/whl/rocm6.2

IMAGE_NAME ?= transcribe
ENV_FILE ?= .env
DOCKER_PORT ?= 8000
MOUNT ?=
GPUS ?= all

BINDIR ?= /usr/bin
CLIENT_BIN := client/transcribe

help:
	@printf "Targets:\n"
	@printf "  make setup           Create venv, upgrade pip, and create runtime dirs\n"
	@printf "  make install-base    Install base deps (CPU path)\n"
	@printf "  make install-nvidia  Install faster-whisper for NVIDIA/CPU\n"
	@printf "  make install-amd     Install ROCm PyTorch + openai-whisper\n"
	@printf "  make install-all     Install both NVIDIA and AMD backends\n"
	@printf "  make check-amd-env   Validate Python/platform for ROCm wheels\n"
	@printf "  make check           Compile-check service modules\n"
	@printf "  make reset-venv      Recreate venv with $(PYTHON_BIN)\n"
	@printf "  make clean           Remove generated cache files\n"
	@printf "  make docker-build    Build the docker image\n"
	@printf "  make docker-run      Run the docker image with an env file\n"
	@printf "  make install-client  Install the transcribe CLI to BINDIR (needs sudo for /usr/bin)\n"
	@printf "  make uninstall-client Remove the transcribe CLI from BINDIR\n"
	@printf "\n"
	@printf "Configurable vars:\n"
	@printf "  PYTHON_BIN=%s\n" "$(PYTHON_BIN)"
	@printf "  ROCM_INDEX=%s\n" "$(ROCM_INDEX)"
	@printf "  IMAGE_NAME=%s\n" "$(IMAGE_NAME)"
	@printf "  ENV_FILE=%s\n" "$(ENV_FILE)"
	@printf "  DOCKER_PORT=%s\n" "$(DOCKER_PORT)"
	@printf "  MOUNT=%s (host dir bound to /workspace, for storage mode=mounted)\n" "$(MOUNT)"
	@printf "  GPUS=%s (--gpus value; set empty to disable GPU passthrough)\n" "$(GPUS)"
	@printf "  BINDIR=%s (install location for the transcribe CLI)\n" "$(BINDIR)"
	@printf "\n"
	@printf "Activate venv in your shell:\n"
	@printf "  source $(VENV)/bin/activate\n"

venv:
	@test -d "$(VENV)" || $(PYTHON_BIN) -m venv "$(VENV)"
	@$(PYTHON) -c 'import sys; req=(3,14); cur=sys.version_info[:2];\
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
	@$(PYTHON) -m py_compile src/transcribe_service/transcribe_core.py src/transcribe_service/api.py

reset-venv:
	@rm -rf "$(VENV)"
	@$(MAKE) setup PYTHON_BIN=$(PYTHON_BIN)

clean:
	@rm -rf __pycache__

docker-build:
	docker build -t $(IMAGE_NAME) -f docker/Dockerfile .

docker-run:
	@test -f "$(ENV_FILE)" || { echo "Env file not found: $(ENV_FILE) (see .env.*.example)" >&2; exit 1; }
	docker run --rm \
		--env-file "$(ENV_FILE)" \
		-p $(DOCKER_PORT):8000 \
		$(if $(GPUS),--gpus $(GPUS),) \
		$(if $(MOUNT),-v "$(MOUNT):/workspace",) \
		$(IMAGE_NAME)

install-client:
	install -m 755 $(CLIENT_BIN) $(BINDIR)/transcribe

uninstall-client:
	rm -f $(BINDIR)/transcribe

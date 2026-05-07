#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# SolQuant — Deployment Script
# ═══════════════════════════════════════════════════════════════════
# Sets up and deploys the three-tier AI stack on Linux Mint.
#
# Prerequisites checked automatically:
#   - Docker Engine
#   - Docker Compose V2
#   - NVIDIA Driver
#   - NVIDIA Container Toolkit
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh              # full deploy
#   ./deploy.sh --build      # force rebuild images
#   ./deploy.sh --down       # tear down stack
#   ./deploy.sh --status     # show stack status
#   ./deploy.sh --logs       # tail all logs
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Constants ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.yml"
PROJECT_NAME="solquant"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helper functions ────────────────────────────────────────────

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[  OK]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[FAIL]${NC}  $*"; }
log_step()  { echo -e "\n${BOLD}═══ $* ═══${NC}"; }

check_command() {
    if command -v "$1" &>/dev/null; then
        log_ok "$1 found: $(command -v "$1")"
        return 0
    else
        log_error "$1 is not installed"
        return 1
    fi
}

# ── Pre-flight checks ──────────────────────────────────────────

preflight_checks() {
    log_step "Pre-flight Checks"
    local failures=0

    # 1. Docker Engine
    if check_command docker; then
        local docker_version
        docker_version=$(docker --version 2>/dev/null || echo "unknown")
        log_info "Docker version: ${docker_version}"

        # Verify Docker daemon is running
        if docker info &>/dev/null; then
            log_ok "Docker daemon is running"
        else
            log_error "Docker daemon is not running. Start it with: sudo systemctl start docker"
            ((failures++))
        fi

        # Check if user can run docker without sudo
        if ! docker ps &>/dev/null; then
            log_warn "Cannot run docker without sudo."
            log_warn "Add your user to the docker group: sudo usermod -aG docker \$USER"
            log_warn "Then log out and back in."
        fi
    else
        log_error "Install Docker: https://docs.docker.com/engine/install/ubuntu/"
        ((failures++))
    fi

    # 2. Docker Compose V2
    if docker compose version &>/dev/null; then
        local compose_version
        compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
        log_ok "Docker Compose V2 found: ${compose_version}"
    else
        log_error "Docker Compose V2 not found."
        log_error "Install: sudo apt-get install docker-compose-plugin"
        ((failures++))
    fi

    # 3. NVIDIA Driver
    if check_command nvidia-smi; then
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total,driver_version \
                   --format=csv,noheader 2>/dev/null || echo "unknown")
        log_info "GPU detected: ${gpu_info}"

        # Parse VRAM to verify it's our target GPU
        local vram_mb
        vram_mb=$(nvidia-smi --query-gpu=memory.total \
                  --format=csv,noheader,nounits 2>/dev/null || echo "0")
        if [ "${vram_mb}" -le 2200 ] 2>/dev/null; then
            log_info "VRAM: ${vram_mb} MB — confirmed low-VRAM GPU (MX550 class)"
        else
            log_warn "VRAM: ${vram_mb} MB — not a 2GB GPU. VRAM budget may need adjustment."
        fi
    else
        log_error "NVIDIA driver not installed."
        log_error "Install: sudo apt install nvidia-driver-560"
        ((failures++))
    fi

    # 4. NVIDIA Container Toolkit
    check_nvidia_container_toolkit
    if [ $? -ne 0 ]; then
        ((failures++))
    fi

    # 5. Verify Docker can access GPU
    if command -v nvidia-smi &>/dev/null && docker info &>/dev/null; then
        log_info "Testing GPU access inside Docker..."
        if docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi &>/dev/null; then
            log_ok "Docker GPU access verified"
        else
            log_error "Docker cannot access GPU. Ensure nvidia-container-toolkit is configured."
            log_error "Run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
            ((failures++))
        fi
    fi

    echo ""
    if [ "${failures}" -gt 0 ]; then
        log_error "${failures} pre-flight check(s) failed. Fix the issues above and re-run."
        exit 1
    fi
    log_ok "All pre-flight checks passed"
}

check_nvidia_container_toolkit() {
    log_info "Checking NVIDIA Container Toolkit..."

    # Method 1: Check if the nvidia runtime is registered with Docker
    if docker info 2>/dev/null | grep -qi "nvidia"; then
        log_ok "NVIDIA runtime registered with Docker"
        return 0
    fi

    # Method 2: Check if the toolkit package is installed
    if dpkg -l | grep -q "nvidia-container-toolkit"; then
        local toolkit_version
        toolkit_version=$(dpkg -l | grep nvidia-container-toolkit | awk '{print $3}')
        log_ok "nvidia-container-toolkit installed: ${toolkit_version}"

        # It's installed but may not be configured
        log_warn "Toolkit is installed but may not be configured as the Docker runtime."
        log_warn "Run: sudo nvidia-ctk runtime configure --runtime=docker"
        log_warn "Then: sudo systemctl restart docker"
        return 0
    fi

    # Method 3: Check for nvidia-container-runtime binary
    if command -v nvidia-container-runtime &>/dev/null; then
        log_ok "nvidia-container-runtime binary found"
        return 0
    fi

    # Not found — provide installation instructions for Linux Mint / Ubuntu
    log_error "NVIDIA Container Toolkit is NOT installed."
    echo ""
    log_info "Install it with the following commands:"
    echo ""
    echo "  # Add NVIDIA container toolkit repo"
    echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\"
    echo "      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo ""
    echo "  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\"
    echo "      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\"
    echo "      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo ""
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo ""
    echo "  # Configure Docker runtime"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    echo ""
    return 1
}

# ── Network setup ───────────────────────────────────────────────

setup_network() {
    log_step "Docker Network Setup"

    local network_name="solquant-net"

    if docker network inspect "${network_name}" &>/dev/null; then
        log_ok "Network '${network_name}' already exists"
    else
        log_info "Creating bridge network '${network_name}'..."
        docker network create \
            --driver bridge \
            --label "com.solquant.project=solquant" \
            "${network_name}"
        log_ok "Network '${network_name}' created"
    fi
}

# ── Volume setup ────────────────────────────────────────────────

setup_volumes() {
    log_step "Docker Volume Setup"

    local volumes=("solquant-model-cache" "solquant-mongo-data" "solquant-alert-logs")

    for vol in "${volumes[@]}"; do
        if docker volume inspect "${vol}" &>/dev/null; then
            log_ok "Volume '${vol}' already exists"
        else
            log_info "Creating volume '${vol}'..."
            docker volume create "${vol}" \
                --label "com.solquant.project=solquant"
            log_ok "Volume '${vol}' created"
        fi
    done
}

# ── Build & Deploy ──────────────────────────────────────────────

build_and_deploy() {
    local build_flag="${1:-}"
    log_step "Building & Deploying Stack"

    cd "${SCRIPT_DIR}"

    # Ensure Maven wrapper exists for the Java build
    ensure_maven_wrapper

    local compose_args="-f ${COMPOSE_FILE} -p ${PROJECT_NAME}"

    if [ "${build_flag}" = "--build" ]; then
        log_info "Forcing image rebuild..."
        docker compose ${compose_args} build --no-cache
    else
        log_info "Building images (cached)..."
        docker compose ${compose_args} build
    fi

    log_info "Starting services..."
    docker compose ${compose_args} up -d

    log_ok "Stack deployed"
}

ensure_maven_wrapper() {
    local mvnw="${SCRIPT_DIR}/agent-controller/mvnw"
    if [ ! -f "${mvnw}" ]; then
        log_info "Maven wrapper not found. Generating..."
        if command -v mvn &>/dev/null; then
            (cd "${SCRIPT_DIR}/agent-controller" && mvn wrapper:wrapper -Dmaven=3.9.9)
        else
            log_info "Maven not installed locally. Downloading wrapper directly..."
            mkdir -p "${SCRIPT_DIR}/agent-controller/.mvn/wrapper"

            local wrapper_url="https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.3.2/maven-wrapper-3.3.2.jar"
            curl -fsSL "${wrapper_url}" \
                -o "${SCRIPT_DIR}/agent-controller/.mvn/wrapper/maven-wrapper.jar"

            # Create wrapper properties
            cat > "${SCRIPT_DIR}/agent-controller/.mvn/wrapper/maven-wrapper.properties" <<'PROPS'
distributionUrl=https://repo.maven.apache.org/maven2/org/apache/maven/apache-maven/3.9.9/apache-maven-3.9.9-bin.zip
wrapperUrl=https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.3.2/maven-wrapper-3.3.2.jar
PROPS

            # Create mvnw script
            curl -fsSL "https://raw.githubusercontent.com/apache/maven-wrapper/master/mvnw" \
                -o "${mvnw}" 2>/dev/null || {
                # Fallback: create a minimal mvnw
                cat > "${mvnw}" <<'MVNW'
#!/bin/sh
exec java -jar "$(dirname "$0")/.mvn/wrapper/maven-wrapper.jar" "$@"
MVNW
            }
            chmod +x "${mvnw}"
        fi
        log_ok "Maven wrapper configured"
    else
        log_ok "Maven wrapper found"
    fi
}

# ── Status ──────────────────────────────────────────────────────

show_status() {
    log_step "Stack Status"

    cd "${SCRIPT_DIR}"
    docker compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" ps

    echo ""
    log_info "Service endpoints:"
    echo "  ├── Inference Engine:  http://localhost:8000  (Swagger: /docs)"
    echo "  ├── Vector DB:         mongodb://localhost:27017"
    echo "  └── Agent Controller:  http://localhost:8081  (Agent: POST /api/agent/chat)"

    echo ""
    log_info "GPU status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader 2>/dev/null || echo "  GPU info unavailable"
}

# ── Teardown ────────────────────────────────────────────────────

teardown() {
    log_step "Tearing Down Stack"

    cd "${SCRIPT_DIR}"
    docker compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" down

    log_ok "Stack stopped and removed"

    echo ""
    log_info "Persistent volumes are preserved:"
    echo "  ├── solquant-model-cache  (downloaded GGUF models)"
    echo "  ├── solquant-mongo-data   (MongoDB data + vector indexes)"
    echo "  └── solquant-alert-logs   (alert log files)"
    log_info "To delete volumes: docker compose -p ${PROJECT_NAME} down -v"
}

# ── Log streaming ──────────────────────────────────────────────

stream_logs() {
    cd "${SCRIPT_DIR}"
    docker compose -f "${COMPOSE_FILE}" -p "${PROJECT_NAME}" logs -f --tail=100
}

# ── Main ────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BOLD}╔════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║     SolQuant AI Stack — Deployment Tool    ║${NC}"
    echo -e "${BOLD}╚════════════════════════════════════════════╝${NC}"
    echo ""

    local action="${1:-deploy}"

    case "${action}" in
        --down|down)
            teardown
            ;;
        --status|status)
            show_status
            ;;
        --logs|logs)
            stream_logs
            ;;
        --build|build)
            preflight_checks
            setup_network
            setup_volumes
            build_and_deploy "--build"
            echo ""
            show_status
            ;;
        deploy|*)
            preflight_checks
            setup_network
            setup_volumes
            build_and_deploy
            echo ""
            show_status
            ;;
    esac

    echo ""
    log_ok "Done."
}

main "$@"

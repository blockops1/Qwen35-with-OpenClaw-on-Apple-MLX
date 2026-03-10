#!/bin/bash
# switch-model.sh — Switch between Qwen3.5 model variants
#
# Usage: switch-model.sh [distilled|instruct]
#
#   distilled — Qwen3.5 Claude Distilled (conversational, reasoning)
#   instruct  — Qwen3.5 Base Instruct (software development, code generation)
#
# Only one vllm-mlx + proxy pair runs at a time (memory constraint).
# This script unloads the current pair and loads the requested one.

set -euo pipefail

TARGET="${1:-}"

if [[ "$TARGET" != "distilled" && "$TARGET" != "instruct" ]]; then
    echo "Usage: switch-model.sh [distilled|instruct]"
    echo ""
    echo "  distilled — Qwen3.5 Claude Distilled (conversational)"
    echo "  instruct  — Qwen3.5 Base Instruct (software development)"
    exit 1
fi

LAUNCHD_DIR="$HOME/Library/LaunchAgents"
GUI_ID="gui/$(id -u)"

# All known vllm and proxy labels (old and new naming)
VLLM_LABELS=(
    "com.user.vllm"
    "com.user.vllm-instruct"
    "com.user.vllm"
)
PROXY_LABELS=(
    "com.user.proxy"
    "com.user.proxy-instruct"
    "com.user.proxy"
)

TARGET_VLLM_LABEL="com.user.vllm-${TARGET}"
TARGET_PROXY_LABEL="com.user.proxy-${TARGET}"
TARGET_VLLM_PLIST="$LAUNCHD_DIR/${TARGET_VLLM_LABEL}.plist"
TARGET_PROXY_PLIST="$LAUNCHD_DIR/${TARGET_PROXY_LABEL}.plist"

echo "=== Switching to: ${TARGET} ==="
if [[ "$TARGET" == "distilled" ]]; then
    echo "    Qwen3.5 Claude Distilled (conversational)"
else
    echo "    Qwen3.5 Base Instruct (software development)"
fi
echo ""

# ── Step 1: Unload all active vllm services ──────────────────────────────────
echo "[1/4] Stopping vllm-mlx..."
for label in "${VLLM_LABELS[@]}"; do
    if launchctl list "$label" &>/dev/null; then
        echo "      Unloading $label"
        launchctl unload "$LAUNCHD_DIR/${label}.plist" 2>/dev/null || true
        sleep 1
    fi
done

# ── Step 2: Unload all active proxy services ──────────────────────────────────
echo "[2/4] Stopping proxy..."
for label in "${PROXY_LABELS[@]}"; do
    if launchctl list "$label" &>/dev/null; then
        echo "      Unloading $label"
        launchctl unload "$LAUNCHD_DIR/${label}.plist" 2>/dev/null || true
        sleep 1
    fi
done

# Brief pause to let processes fully exit before loading new ones
sleep 2

# ── Step 3: Load target vllm plist ───────────────────────────────────────────
echo "[3/4] Starting vllm-mlx (${TARGET})..."
if [[ ! -f "$TARGET_VLLM_PLIST" ]]; then
    echo "ERROR: plist not found: $TARGET_VLLM_PLIST"
    exit 1
fi
launchctl load "$TARGET_VLLM_PLIST"
sleep 1

# ── Step 4: Load target proxy plist ──────────────────────────────────────────
echo "[4/4] Starting proxy (${TARGET})..."
if [[ ! -f "$TARGET_PROXY_PLIST" ]]; then
    echo "ERROR: plist not found: $TARGET_PROXY_PLIST"
    exit 1
fi
launchctl load "$TARGET_PROXY_PLIST"
sleep 2

# ── Confirm ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Status check ==="
if launchctl list "$TARGET_VLLM_LABEL" &>/dev/null; then
    echo "✅  vllm-mlx  ($TARGET_VLLM_LABEL) — running"
else
    echo "❌  vllm-mlx  ($TARGET_VLLM_LABEL) — NOT running (check log)"
fi

if launchctl list "$TARGET_PROXY_LABEL" &>/dev/null; then
    echo "✅  proxy     ($TARGET_PROXY_LABEL) — running"
else
    echo "❌  proxy     ($TARGET_PROXY_LABEL) — NOT running (check log)"
fi

echo ""
echo "Note: vllm-mlx will take 2–5 min to load the model."
echo "      Watch: tail -f ~/mlx-server/vllm-server.log"
echo ""
echo "Done. Next Telegram session will announce: Qwen3.5 $([ "$TARGET" = "distilled" ] && echo "Claude Distilled (conversational)" || echo "Base Instruct (software development)")"

#!/bin/bash
# record_meeting.sh — Capture desktop audio + microphone into a Whisper-ready WAV file.
#
# Usage:
#   ./record_meeting.sh                    # auto-named  meeting_YYYYMMDD_HHMM.wav
#   ./record_meeting.sh my_standup.wav     # custom output name

OUTPUT_FILE="${1:-meeting_$(date +%Y%m%d_%H%M%S).wav}"

echo "🔍 Searching for PipeWire/PulseAudio devices..."

# Auto-detect the default sink monitor (system/desktop audio) and mic
DESKTOP_SINK=$(pactl info | grep "Default Sink" | awk '{print $3}').monitor
MIC_SOURCE=$(pactl info | grep "Default Source" | awk '{print $3}')

if [ -z "$DESKTOP_SINK" ] || [ -z "$MIC_SOURCE" ]; then
    echo "❌ Error: Could not automatically detect audio devices."
    echo "   Run 'pactl list short sources' to inspect available sources manually."
    exit 1
fi

# Guard against the Dummy Output bug present in some Pop!_OS 24.04 / kernel 6.16+ installs.
# If PipeWire loses track of the hardware device it falls back to a null sink named
# 'auto_null' or 'dummy', which would produce a silent recording with no warning.
if echo "$DESKTOP_SINK $MIC_SOURCE" | grep -qiE "dummy|auto_null"; then
    echo "❌ Error: A dummy/null audio device was detected — your PipeWire session has"
    echo "   lost track of the hardware. Reset it with:"
    echo ""
    echo "   systemctl --user restart wireplumber pipewire pipewire-pulse"
    echo ""
    echo "   Then re-run this script. If the problem persists, check kernel version with"
    echo "   'uname -r' — kernel 6.16.x has a known HDA audio regression on Pop!_OS 24.04."
    exit 1
fi

echo "🎙️  Microphone   : $MIC_SOURCE"
echo "🔊  System audio : $DESKTOP_SINK"
echo "💾  Output file  : $OUTPUT_FILE"
echo "----------------------------------------------------"
echo "🛑  Press [CTRL+C] to stop recording."
echo "----------------------------------------------------"

# Record both streams simultaneously, mix them down to a single 16 kHz mono WAV.
# 16 kHz / mono is Whisper's native format — keeping it this way avoids any
# lossy resampling step at transcription time.
ffmpeg -loglevel warning \
    -f pulse -i "$DESKTOP_SINK" \
    -f pulse -i "$MIC_SOURCE" \
    -filter_complex "amix=inputs=2:duration=longest:dropout_transition=2" \
    -ac 1 -ar 16000 \
    "$OUTPUT_FILE"

echo -e "\n💾 Recording saved → $OUTPUT_FILE"

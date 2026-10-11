#!/bin/bash
# ============================================================================
# Pi Scanner Watchdog
# ============================================================================
# Runs every minute (via systemd timer). If the 1-minute load average stays
# critically high for several consecutive checks, it restarts the transcriber
# (the usual CPU hog) to let the system recover — so the Pi never needs a
# manual power cycle again.
#
# State is kept in /run (tmpfs) so it resets on reboot.
# ============================================================================

STATE=/run/scanner_watchdog_strikes
NPROC=$(nproc)
# Critical = sustained load > 1.5x core count (e.g. >6.0 on a 4-core Pi)
CRIT=$(awk "BEGIN{print $NPROC*1.5}")

load1=$(awk '{print $1}' /proc/loadavg)
strikes=$(cat "$STATE" 2>/dev/null || echo 0)

over=$(awk "BEGIN{print ($load1 > $CRIT) ? 1 : 0}")

if [ "$over" = "1" ]; then
    strikes=$((strikes + 1))
else
    strikes=0
fi
echo "$strikes" > "$STATE"

logger -t scanner-watchdog "load1=$load1 crit=$CRIT strikes=$strikes"

# 3 consecutive minutes over critical load -> intervene
if [ "$strikes" -ge 3 ]; then
    logger -t scanner-watchdog "CRITICAL: load $load1 for ${strikes}min, restarting pi-transcriber"
    systemctl restart pi-transcriber.service
    echo 0 > "$STATE"
fi

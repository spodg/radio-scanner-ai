#!/bin/bash
# Ensure NAS is mounted before services start.
# Retries up to 10 times with 3s delay (covers slow network on boot).
for i in 1 2 3 4 5 6 7 8 9 10; do
    if ls /mnt/nas/ >/dev/null 2>&1 && mountpoint -q /mnt/nas; then
        exit 0
    fi
    umount -l /mnt/nas 2>/dev/null
    mount /mnt/nas 2>/dev/null
    sleep 3
done
# Last attempt
mount /mnt/nas 2>/dev/null
mountpoint -q /mnt/nas

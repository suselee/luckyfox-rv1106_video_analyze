#!/bin/sh

APP_DIR=/root/daughter_watch
BIN="$APP_DIR/daughter_watch"
CONFIG="$APP_DIR/config.ini"
LOG=/tmp/daughter_watch.log
MAX_LOG_BYTES=5242880
MIN_VALID_YEAR=2025
TIME_SYNC_RETRY_SECONDS=5

export LD_LIBRARY_PATH="$APP_DIR:/oem/usr/lib:${LD_LIBRARY_PATH}"
cd "$APP_DIR" || exit 1

child=""
stop_child() {
    if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
        kill "$child" 2>/dev/null
        wait "$child" 2>/dev/null
    fi
    exit 0
}
trap stop_child INT TERM

current_year() {
    year=$(date +%Y 2>/dev/null)
    case "$year" in
        ''|*[!0-9]*) echo 0 ;;
        *) echo "$year" ;;
    esac
}

while [ "$(current_year)" -lt "$MIN_VALID_YEAR" ]; do
    echo "[time-sync] invalid clock; forcing NTP sync" >>"$LOG"
    [ -x /etc/init.d/S49ntp ] && /etc/init.d/S49ntp stop >>"$LOG" 2>&1
    /usr/sbin/ntpd -4 -g -G -q >>"$LOG" 2>&1 || true
    [ -x /etc/init.d/S49ntp ] && /etc/init.d/S49ntp start >>"$LOG" 2>&1
    [ "$(current_year)" -ge "$MIN_VALID_YEAR" ] && break
    sleep "$TIME_SYNC_RETRY_SECONDS"
done

while true; do
    if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
        mv -f "$LOG" "$LOG.1"
    fi
    "$BIN" "$CONFIG" >>"$LOG" 2>&1 &
    child=$!
    wait "$child"
    rc=$?
    child=""
    echo "[supervisor] daughter_watch exited rc=$rc; retrying in 5s" >>"$LOG"
    sleep 5
done

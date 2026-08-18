#!/bin/sh

APP_DIR=/root/daughter_watch
BIN="$APP_DIR/daughter_watch"
CONFIG="$APP_DIR/config.ini"
LOG=/tmp/daughter_watch.log
MAX_LOG_BYTES=5242880
MIN_VALID_YEAR=2025
TIME_SYNC_RETRY_SECONDS=5
FFMPEG=/root/ffmpeg
# RTSP 凭据按现场填写 (与 config.ini 一致); 仓库内脱敏
RTSP_URL='rtsp://admin:CHANGE_ME@<cam-ip>:554/stream2'
RTSP_HIGH_URL='rtsp://admin:CHANGE_ME@<cam-ip>:554/stream1'
HIGH_FIFO=/tmp/high4k.fifo
HIGH_AUDIO_FIFO=/tmp/high4k_audio.fifo
STDIN_FIFO=/tmp/stream2.fifo
MONITOR_INTERVAL=15

export LD_LIBRARY_PATH="$APP_DIR:/oem/usr/lib:${LD_LIBRARY_PATH}"
cd "$APP_DIR" || exit 1

child=""
high_pid=""
ff_stdin_pid=""
monitor_pid=""

stop_child() {
    [ -n "$monitor_pid" ] && kill "$monitor_pid" 2>/dev/null
    [ -n "$child" ] && kill "$child" 2>/dev/null && wait "$child" 2>/dev/null
    killall ffmpeg 2>/dev/null
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

monitor_pipeline() {
    while true; do
        sleep "$MONITOR_INTERVAL"
        if ! kill -0 "$high_pid" 2>/dev/null || ! kill -0 "$ff_stdin_pid" 2>/dev/null; then
            echo "[supervisor] ffmpeg died (high=$high_pid stdin=$ff_stdin_pid); restarting pipeline" >>"$LOG"
            [ -n "$child" ] && kill "$child" 2>/dev/null
            return 0
        fi
        if ! kill -0 "$child" 2>/dev/null; then
            return 0
        fi
    done
}

while true; do
    if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
        mv -f "$LOG" "$LOG.1"
    fi
    # FIFO open 是同步握手, 写端先 open 会阻塞等读端, 消除管道未建立竞态
    rm -f "$HIGH_FIFO" "$HIGH_AUDIO_FIFO" "$STDIN_FIFO"
    mkfifo "$HIGH_FIFO" 2>/dev/null
    mkfifo "$HIGH_AUDIO_FIFO" 2>/dev/null
    mkfifo "$STDIN_FIFO" 2>/dev/null
    # 4K + 音频: stream1 -> 双 FIFO (high_stream 窗口外关闭读端时写阻塞休眠)
    "$FFMPEG" -rtsp_transport tcp -i "$RTSP_HIGH_URL" \
        -map 0:v -c:v copy -f hevc -bsf:v hevc_mp4toannexb -y "$HIGH_FIFO" \
        -map 0:a -c:a copy -f adts -y "$HIGH_AUDIO_FIFO" 2>>"$LOG" &
    high_pid=$!
    # 检测流: stream2 -> FIFO -> daughter_watch stdin
    "$FFMPEG" -rtsp_transport tcp -i "$RTSP_URL" -map 0:v -c:v copy \
        -f h264 -bsf:v h264_mp4toannexb - 2>>"$LOG" > "$STDIN_FIFO" &
    ff_stdin_pid=$!
    "$BIN" "$CONFIG" < "$STDIN_FIFO" >>"$LOG" 2>&1 &
    child=$!
    # 监控: 任一 ffmpeg 死亡 -> 杀掉 daughter_watch 触发重启
    monitor_pipeline &
    monitor_pid=$!
    wait "$child"
    rc=$?
    kill "$monitor_pid" 2>/dev/null
    child=""
    kill "$high_pid" "$ff_stdin_pid" 2>/dev/null
    echo "[supervisor] pipeline exited rc=$rc; retrying in 5s" >>"$LOG"
    sleep 5
done

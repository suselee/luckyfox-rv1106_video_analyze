#!/bin/sh

APP_DIR=/root/daughter_watch
BIN="$APP_DIR/daughter_watch"
CONFIG="$APP_DIR/config.ini"
LOG=/tmp/daughter_watch.log
MAX_LOG_BYTES=10485760
# 日志保留两轮 (~2-3 天, /tmp 为 tmpfs 不伤闪存):
# 断线类故障的取证高度依赖完整日志, 5MB 单轮只保 ~1 天曾导致现场丢失。
MIN_VALID_YEAR=2025
TIME_SYNC_RETRY_SECONDS=5
FFMPEG=/root/ffmpeg
HIGH_FIFO=/tmp/high4k.fifo
HIGH_AUDIO_FIFO=/tmp/high4k_audio.fifo
STDIN_FIFO=/tmp/stream2.fifo
MONITOR_INTERVAL=15
# 熔断退避: 快失败(<60s)指数拉长重启间隔; 存活 >=600s 视为健康清零
FAST_FAIL_SECONDS=60
HEALTHY_RUN_SECONDS=600
BACKOFF_BASE=30
BACKOFF_MAX=900
# RTSP 地址单一事实源是 config.ini ([rtsp]url / [high]url);
# 下面仅为 config.ini 缺失/损坏时的兜底 (仓库内脱敏)
RTSP_URL_DEFAULT='rtsp://admin:CHANGE_ME@<cam-ip>:554/stream2'
RTSP_HIGH_URL_DEFAULT='rtsp://admin:CHANGE_ME@<cam-ip>:554/stream1'

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

# 从 ini 取值: cfg_val <file> <section> <key>
cfg_val() {
    awk -v sec="[$2]" -v key="$3" '
        $0 == sec {insec=1; next}
        /^\[/ {insec=0}
        insec && $1 == key {sub(/^[^=]*=[[:space:]]*/, ""); print; exit}
    ' "$1" 2>/dev/null
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
        # 数据活度 (兜底): 三进程全活但无数据是曾经的永久假活形态。
        # /tmp/dw_data 由 daughter_watch 窗口内每次 HEALTH(60s)刷新,
        # 内容为最后收到码流的时刻; 窗口外 sleeping 不写, 故只在窗口内检查。
        if data_stale; then
            echo "[supervisor] no stream data for 10+ min (all processes alive); restarting pipeline" >>"$LOG"
            [ -n "$child" ] && kill "$child" 2>/dev/null
            return 0
        fi
    done
}

data_stale() {
    # 窗口 07:00-21:00 UTC+8 对应 23:00-13:00 UTC。
    h=$(date -u +%H 2>/dev/null)
    m=$(date -u +%M 2>/dev/null)
    case "$h" in
        ''|*[!0-9]*) return 1 ;;
    esac
    # 去掉前导 0, 避免 sh 八进制解释 (如 08/09)。
    h=${h#0}; m=${m#0}
    [ -z "$h" ] && h=0; [ -z "$m" ] && m=0
    in_window=0
    if [ "$h" -ge 23 ] || [ "$h" -lt 13 ]; then in_window=1; fi
    # 窗口开始后 15 分钟宽限 (冷启动/相机恢复/退避需要时间)。
    if [ "$h" -eq 23 ] && [ "$m" -lt 15 ]; then in_window=0; fi
    if [ "$in_window" -eq 0 ]; then return 1; fi
    if [ -f /tmp/dw_data ]; then
        # 心跳超 10 分钟没刷新 → 主循环已无数据 (C++ 内 240s 熔断应先触发,
        # 这里是它失效/卡死在 HEALTH 之外的最后一道防线)。
        if find /tmp/dw_data -mmin +10 2>/dev/null | grep -q .; then
            return 0
        fi
    else
        # 进程起来超过 15 分钟还没有心跳文件 → 主循环根本没转起来。
        if [ -n "$child" ] && find "/proc/$child" -mmin +15 2>/dev/null | grep -q .; then
            return 0
        fi
    fi
    return 1
}

fail_count=0
while true; do
    if [ -f "$LOG" ] && [ "$(wc -c < "$LOG")" -gt "$MAX_LOG_BYTES" ]; then
        mv -f "$LOG.1" "$LOG.2" 2>/dev/null
        mv -f "$LOG" "$LOG.1"
    fi
    RTSP_URL=$(cfg_val "$CONFIG" rtsp url)
    RTSP_HIGH_URL=$(cfg_val "$CONFIG" high url)
    [ -n "$RTSP_URL" ] || RTSP_URL="$RTSP_URL_DEFAULT"
    [ -n "$RTSP_HIGH_URL" ] || RTSP_HIGH_URL="$RTSP_HIGH_URL_DEFAULT"
    # FIFO open 是同步握手, 写端先 open 会阻塞等读端, 消除管道未建立竞态。
    # 注意: 读端由 daughter_watch 全程持有 (窗口外只暂停读取), 绝不关闭,
    # 否则写端 ffmpeg 会因 SIGPIPE 死亡并触发整条管线重启循环。
    rm -f "$HIGH_FIFO" "$HIGH_AUDIO_FIFO" "$STDIN_FIFO"
    mkfifo "$HIGH_FIFO" 2>/dev/null
    mkfifo "$HIGH_AUDIO_FIFO" 2>/dev/null
    mkfifo "$STDIN_FIFO" 2>/dev/null
    # 4K + 音频: stream1 -> 双 FIFO。-loglevel warning -nostats 抑制进度刷屏。
    # -timeout 10s: RTSP socket I/O 超时 (微秒, 建连/读卡死时退出,
    # 由 supervisor 重建; 板端 ffmpeg-7.0.2-static 不支持 -reconnect*/-stimeout,
    # 加了会直接 "Option not found" 退出 → 切勿加)。
    # 注意绝不能加 -rw_timeout: 窗口外 daughter_watch 暂停读取、靠背压让
    # ffmpeg 休眠, 输出超时会把这种正常休眠误杀成整夜重启循环。
    "$FFMPEG" -loglevel warning -nostats -rtsp_transport tcp -timeout 10000000 \
        -i "$RTSP_HIGH_URL" \
        -map 0:v -c:v copy -f hevc -bsf:v hevc_mp4toannexb -y "$HIGH_FIFO" \
        -map 0:a -c:a copy -f adts -y "$HIGH_AUDIO_FIFO" 2>>"$LOG" &
    high_pid=$!
    # 检测流: stream2 -> FIFO -> daughter_watch stdin
    "$FFMPEG" -loglevel warning -nostats -rtsp_transport tcp -timeout 10000000 \
        -i "$RTSP_URL" \
        -map 0:v -c:v copy -f h264 -bsf:v h264_mp4toannexb - \
        2>>"$LOG" > "$STDIN_FIFO" &
    ff_stdin_pid=$!
    "$BIN" "$CONFIG" < "$STDIN_FIFO" >>"$LOG" 2>&1 &
    child=$!
    run_start=$(date +%s)
    # 监控: 任一 ffmpeg 死亡 -> 杀掉 daughter_watch 触发重启
    monitor_pipeline &
    monitor_pid=$!
    wait "$child"
    rc=$?
    kill "$monitor_pid" 2>/dev/null
    monitor_pid=""
    child=""
    kill "$high_pid" "$ff_stdin_pid" 2>/dev/null
    ran=$(( $(date +%s) - run_start ))
    if [ "$ran" -lt "$FAST_FAIL_SECONDS" ]; then
        fail_count=$((fail_count + 1))
        delay=$BACKOFF_BASE
        i=1
        while [ "$i" -lt "$fail_count" ]; do
            delay=$((delay * 2))
            [ "$delay" -ge "$BACKOFF_MAX" ] && { delay=$BACKOFF_MAX; break; }
            i=$((i + 1))
        done
        echo "[supervisor] pipeline exited rc=$rc fast-fail x$fail_count (ran=${ran}s); backing off ${delay}s" >>"$LOG"
    else
        if [ "$ran" -ge "$HEALTHY_RUN_SECONDS" ]; then
            fail_count=0
        fi
        delay=5
        echo "[supervisor] pipeline exited rc=$rc (ran=${ran}s); retrying in ${delay}s" >>"$LOG"
    fi
    sleep "$delay"
done

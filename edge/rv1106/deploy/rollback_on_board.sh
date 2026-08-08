#!/bin/sh
set -eu

APP_DIR=/root/daughter_watch
INIT_SCRIPT=/etc/init.d/S98daughter_watch
ROLLBACK_DIR=/root/daughter_watch.rollback

[ -d "$ROLLBACK_DIR" ] || { echo "rollback snapshot not found" >&2; exit 1; }
"$INIT_SCRIPT" stop || true
for path in daughter_watch config.ini run.sh; do
    [ -e "$ROLLBACK_DIR/$path" ] && cp -p "$ROLLBACK_DIR/$path" "$APP_DIR/$path"
done
[ -e "$ROLLBACK_DIR/daughter.db" ] && cp -p "$ROLLBACK_DIR/daughter.db" "$APP_DIR/models/daughter.db"
[ -e "$ROLLBACK_DIR/S98daughter_watch" ] && cp -p "$ROLLBACK_DIR/S98daughter_watch" "$INIT_SCRIPT"
chmod +x "$APP_DIR/daughter_watch" "$APP_DIR/run.sh" "$INIT_SCRIPT"
"$INIT_SCRIPT" start
echo "rollback complete"

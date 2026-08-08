#!/bin/sh
set -eu

PACKAGE_DIR=${1:-.}
APP_DIR=/root/daughter_watch
INIT_SCRIPT=/etc/init.d/S98daughter_watch
ROLLBACK_DIR=/root/daughter_watch.rollback

required="daughter_watch rockiva_probe track_fusion_test schedule_test librockiva.so models/face_detector.rknn models/mobilefacenet.rknn models/daughter.db models/rockiva/object_detection_pfp.data"
for path in $required; do
    [ -e "$PACKAGE_DIR/$path" ] || {
        echo "missing package file: $PACKAGE_DIR/$path" >&2
        exit 1
    }
done

"$INIT_SCRIPT" stop || true
rm -rf "$ROLLBACK_DIR"
mkdir -p "$ROLLBACK_DIR"
for path in daughter_watch config.ini run.sh; do
    [ -e "$APP_DIR/$path" ] && cp -p "$APP_DIR/$path" "$ROLLBACK_DIR/$path"
done
[ -e "$APP_DIR/models/daughter.db" ] && cp -p "$APP_DIR/models/daughter.db" "$ROLLBACK_DIR/daughter.db"
[ -e "$INIT_SCRIPT" ] && cp -p "$INIT_SCRIPT" "$ROLLBACK_DIR/S98daughter_watch"

mkdir -p "$APP_DIR/models/rockiva"
cp -p "$PACKAGE_DIR/rockiva_probe" "$PACKAGE_DIR/track_fusion_test" "$PACKAGE_DIR/schedule_test" "$APP_DIR/"
cp -p "$PACKAGE_DIR/librockiva.so" "$APP_DIR/"
cp -p "$PACKAGE_DIR/rollback_on_board.sh" "$APP_DIR/rollback_on_board.sh"
cp -p "$PACKAGE_DIR/models/rockiva/object_detection_pfp.data" "$APP_DIR/models/rockiva/"
cp -p "$PACKAGE_DIR/models/face_detector.rknn" "$PACKAGE_DIR/models/mobilefacenet.rknn" "$APP_DIR/models/"
[ -e "$APP_DIR/models/daughter.db" ] || cp -p "$PACKAGE_DIR/models/daughter.db" "$APP_DIR/models/daughter.db"
chmod +x "$APP_DIR/rockiva_probe" "$APP_DIR/track_fusion_test" "$APP_DIR/schedule_test" "$APP_DIR/rollback_on_board.sh"

export LD_LIBRARY_PATH="$APP_DIR:/oem/usr/lib:${LD_LIBRARY_PATH:-}"
"$APP_DIR/track_fusion_test"
"$APP_DIR/schedule_test"
if ! "$APP_DIR/rockiva_probe" "$APP_DIR/config.ini" 20; then
    echo "RockIVA probe failed; leaving the old production binary in place" >&2
    "$INIT_SCRIPT" start || true
    exit 2
fi

cp -p "$PACKAGE_DIR/daughter_watch" "$APP_DIR/daughter_watch.new"
chmod +x "$APP_DIR/daughter_watch.new"
mv -f "$APP_DIR/daughter_watch.new" "$APP_DIR/daughter_watch"
[ -e "$PACKAGE_DIR/run.sh" ] && cp -p "$PACKAGE_DIR/run.sh" "$APP_DIR/run.sh"
[ -e "$PACKAGE_DIR/S98daughter_watch" ] && cp -p "$PACKAGE_DIR/S98daughter_watch" "$INIT_SCRIPT"
chmod +x "$APP_DIR/daughter_watch" "$APP_DIR/run.sh" "$INIT_SCRIPT"
"$INIT_SCRIPT" start
echo "installed; rollback with /root/daughter_watch/rollback_on_board.sh"

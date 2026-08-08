// 女儿底库注册工具 (RV1106, 与主服务同一套模型)
//
// 输入若干张女儿照片, 对每张检测最大的人脸, 提取特征, 累加进底库文件。
// 用法:
//   ./enroll <detector.rknn> <recognizer.rknn> <out.db> <img1> [img2 ...]
//   ./enroll --append <detector.rknn> <recognizer.rknn> <out.db> <img...>   # 在已有底库上追加
//
// 建议 5~20 张, 覆盖正/侧脸、明/暗光、不同发型。婴儿脸变化快, 每 1~2 月补拍更新。

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#include "stb/stb_image.h"

#include "facedb.h"
#include "face_detect.h"
#include "face_recog.h"

using namespace dw;

// 选面积最大的人脸(注册照通常主体就是女儿, 取最大脸最稳)
static int largest_face(const std::vector<FaceBox>& faces) {
    int   best = -1;
    float area = 0;
    for (size_t i = 0; i < faces.size(); i++) {
        float a = (faces[i].x2 - faces[i].x1) * (faces[i].y2 - faces[i].y1);
        if (a > area) { area = a; best = (int)i; }
    }
    return best;
}

int main(int argc, char* argv[]) {
    int argi = 1;
    bool append = false;
    if (argc > argi && strcmp(argv[argi], "--append") == 0) { append = true; argi++; }

    if (argc - argi < 4) {
        printf("Usage: %s [--append] <detector.rknn> <recognizer.rknn> <out.db> <img1> [img2 ...]\n", argv[0]);
        return 1;
    }

    const char* det_path = argv[argi++];
    const char* rec_path = argv[argi++];
    const char* db_path  = argv[argi++];

    FaceDetector detector;
    if (!detector.init(det_path)) { printf("[ERR] detector init failed\n"); return 1; }
    FaceRecognizer recognizer;
    if (!recognizer.init(rec_path)) { printf("[ERR] recognizer init failed\n"); return 1; }

    FaceDB db;
    if (append) {
        if (db.load(db_path)) printf("[INFO] 追加模式: 已有 %d 条\n", db.count());
        else                  printf("[INFO] 追加模式: 无已有底库, 新建\n");
    }

    int added = 0;

    for (; argi < argc; argi++) {
        const char* img_path = argv[argi];
        int w, h, c;
        unsigned char* rgb = stbi_load(img_path, &w, &h, &c, 3);
        if (!rgb) { printf("[SKIP] 无法读取: %s\n", img_path); continue; }

        std::vector<FaceBox> faces = detector.detect(rgb, w, h, 0.6f);
        int idx = largest_face(faces);
        if (idx < 0) {
            printf("[SKIP] 未检测到人脸: %s\n", img_path);
            stbi_image_free(rgb);
            continue;
        }

        std::vector<float> emb;
        if (!recognizer.extract(rgb, w, h, faces[idx], emb)) {
            printf("[SKIP] 特征提取失败: %s\n", img_path);
            stbi_image_free(rgb);
            continue;
        }
        stbi_image_free(rgb);

        db.add(emb);
        added++;
        printf("[OK] %s (face %d/%d, score=%.2f)\n",
               img_path, idx + 1, (int)faces.size(), faces[idx].score);
    }

    if (added == 0) { printf("[ERR] 没有任何有效人脸, 未写入底库\n"); return 1; }

    if (!db.save(db_path)) { printf("[ERR] 保存失败: %s\n", db_path); return 1; }
    printf("\n[DONE] 底库已保存: %s (共 %d 条, dim=%d)\n", db_path, db.count(), db.dim());

    detector.destroy();
    recognizer.destroy();
    return 0;
}

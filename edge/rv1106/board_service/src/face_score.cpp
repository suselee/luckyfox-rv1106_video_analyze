// 底库相似度打分工具 (RV1106, 与主服务/enroll 同一套模型)
//
// 用途: 批量扩底库前先做身份过滤 —— 对每张候选图检测最大人脸、提特征,
//       与现有 daughter.db 比对, 只保留相似度达标(确认是女儿)的样本。
//       enroll 本身是"无脑追加", 直接灌入会把大人脸也写进底库。
//
// 用法:
//   ./face_score <detector.rknn> <recognizer.rknn> <db> <img1> [img2 ...]
//
// 输出 (每行一个样本, 便于 awk 过滤):
//   OK   <sim>\t<img>
//   NOFACE\t<img>
//   ERR  \t<img>
// 退出码: 0 正常跑完; 1 参数/初始化错误

#include <stdio.h>
#include <string.h>
#include <vector>
#include <string>

#include "stb/stb_image.h"

#include "facedb.h"
#include "face_detect.h"
#include "face_recog.h"

using namespace dw;

static int largest_face(const std::vector<FaceBox>& faces) {
    int best = -1;
    float area = 0;
    for (size_t i = 0; i < faces.size(); i++) {
        float a = (faces[i].x2 - faces[i].x1) * (faces[i].y2 - faces[i].y1);
        if (a > area) { area = a; best = (int)i; }
    }
    return best;
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        printf("Usage: %s <detector.rknn> <recognizer.rknn> <db> <img1> [img2 ...]\n", argv[0]);
        return 1;
    }
    const char* det_path = argv[1];
    const char* rec_path = argv[2];
    const char* db_path  = argv[3];

    FaceDetector detector;
    if (!detector.init(det_path)) { printf("[ERR] detector init failed\n"); return 1; }
    FaceRecognizer recognizer;
    if (!recognizer.init(rec_path)) { printf("[ERR] recognizer init failed\n"); return 1; }
    FaceDB db;
    if (!db.load(db_path)) { printf("[ERR] db load failed: %s\n", db_path); return 1; }
    printf("[INFO] 底库 %d 条, dim=%d\n", db.count(), db.dim());

    for (int i = 4; i < argc; i++) {
        const char* img_path = argv[i];
        int w = 0, h = 0, c = 0;
        unsigned char* rgb = stbi_load(img_path, &w, &h, &c, 3);
        if (!rgb) { printf("ERR\t%s\n", img_path); continue; }

        std::vector<FaceBox> faces = detector.detect(rgb, w, h, 0.6f);
        int idx = largest_face(faces);
        if (idx < 0) {
            printf("NOFACE\t%s\n", img_path);
            stbi_image_free(rgb);
            continue;
        }
        std::vector<float> emb;
        bool ok = recognizer.extract(rgb, w, h, faces[idx], emb);
        stbi_image_free(rgb);
        if (!ok) { printf("ERR\t%s\n", img_path); continue; }

        float sim = db.best_similarity(emb);
        int fw = (int)((faces[idx].x2 - faces[idx].x1) * w);
        int fh = (int)((faces[idx].y2 - faces[idx].y1) * h);
        printf("OK\t%.4f\t%dx%d\t%s\n", sim, fw, fh, img_path);
    }
    return 0;
}

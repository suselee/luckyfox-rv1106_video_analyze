#include "facedb.h"
#include <cstdio>
#include <cmath>
#include <cstring>

namespace dw {

void FaceDB::l2_normalize(std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += static_cast<double>(x) * x;
    float n = static_cast<float>(std::sqrt(s));
    if (n > 1e-12f)
        for (float& x : v) x /= n;
}

void FaceDB::add(const std::vector<float>& feat) {
    if (feats_.empty() && dim_ == 0)
        dim_ = static_cast<int>(feat.size());
    if (static_cast<int>(feat.size()) != dim_ || dim_ <= 0)
        return; // 维度不符, 忽略
    std::vector<float> f = feat;
    l2_normalize(f);
    feats_.push_back(std::move(f));
}

float FaceDB::best_similarity(const std::vector<float>& query) const {
    if (feats_.empty() || static_cast<int>(query.size()) != dim_)
        return -1.0f;
    std::vector<float> q = query;
    l2_normalize(q);
    float best = -1.0f;
    for (const auto& f : feats_) {
        float dot = 0.0f;
        for (int i = 0; i < dim_; ++i) dot += q[i] * f[i];
        if (dot > best) best = dot;
    }
    return best;
}

bool FaceDB::load(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    char magic[4] = {0};
    int32_t dim = 0, count = 0;
    bool ok = std::fread(magic, 1, 4, fp) == 4 &&
              std::memcmp(magic, "FDB1", 4) == 0 &&
              std::fread(&dim, sizeof(int32_t), 1, fp) == 1 &&
              std::fread(&count, sizeof(int32_t), 1, fp) == 1 &&
              dim > 0 && count >= 0;

    if (ok) {
        feats_.clear();
        dim_ = dim;
        for (int i = 0; i < count && ok; ++i) {
            std::vector<float> f(dim);
            ok = std::fread(f.data(), sizeof(float), dim, fp) == static_cast<size_t>(dim);
            if (ok) feats_.push_back(std::move(f));
        }
    }
    std::fclose(fp);
    return ok;
}

bool FaceDB::save(const std::string& path) const {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;

    int32_t dim = dim_, count = static_cast<int32_t>(feats_.size());
    bool ok = std::fwrite("FDB1", 1, 4, fp) == 4 &&
              std::fwrite(&dim, sizeof(int32_t), 1, fp) == 1 &&
              std::fwrite(&count, sizeof(int32_t), 1, fp) == 1;
    for (int i = 0; i < count && ok; ++i)
        ok = std::fwrite(feats_[i].data(), sizeof(float), dim, fp) == static_cast<size_t>(dim);

    std::fclose(fp);
    return ok;
}

} // namespace dw

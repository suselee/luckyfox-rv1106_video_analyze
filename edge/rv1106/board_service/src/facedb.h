#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace dw {

// 女儿底库 (face database)。
//
// 文件格式 daughter.db (little-endian):
//   magic : 4 bytes  "FDB1"
//   dim   : int32     特征维度 (如 512)
//   count : int32     底库中特征条数
//   data  : count * dim * float32   每条已 L2 归一化的特征向量
//
// 约定: 所有向量存入前一律 L2 归一化, 于是余弦相似度 == 点积, 比对时只需做内积。
class FaceDB {
public:
    bool load(const std::string& path);
    bool save(const std::string& path) const;

    // 追加一条特征 (内部自动 L2 归一化)。
    // 空库时用首条确定 dim; 之后维度不符的条目被忽略。
    void add(const std::vector<float>& feat);

    // 返回 query 与底库中最相似一条的余弦相似度, 取值 [-1,1]。
    // 空库或维度不符返回 -1。query 内部按需归一化, 不改动入参。
    float best_similarity(const std::vector<float>& query) const;

    int  dim()   const { return dim_; }
    int  count() const { return static_cast<int>(feats_.size()); }
    bool empty() const { return feats_.empty(); }

    static void l2_normalize(std::vector<float>& v);

private:
    int dim_ = 0;
    std::vector<std::vector<float>> feats_; // 每条均已归一化
};

} // namespace dw

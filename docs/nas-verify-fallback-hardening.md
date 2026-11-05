# NAS 端 probable 整帧回退的收紧说明

> 2026-11-05 · 针对"只有大人的片段被保存"的 NAS 侧加固
> 相关代码：`src/nas_video_summarizer/ingest.py::_verify_probable_clip`

## 1. 现象

2026-11-05 落盘的 8 条 probable 片段里，有 2 条画面中**只有大人**：

| 片段 | 时间 | 画面 | 判定 |
|---|---|---|---|
| 3784 | 07:05 | 只有爸爸坐在沙发 | 误报 |
| 3791 | 10:04 | 只有奶奶往洗衣机倒洗衣液 | 误报 |

## 2. 板端根因（已修复，供参考）

`track_fusion.cpp` 的 `relative_child` 旧逻辑只比较"比画面最高的人矮 65%"，
**没有最小尺寸约束**，于是紧贴大人的小目标误检框（玩具/桌面杂物，实测
`13x29px`、归一化高 0.08–0.10）被判成"儿童"，升 probable 并切片上传。

已修复：`relative_child` 增加 `height >= 0.12f` 下限（commit `228060f`）。

## 3. NAS 侧为什么仍会放行

板端框不可靠时，ROI 校验必然 `no_person`，于是走**整帧回退**：

```python
# ingest.py:446-478
if not verification.accepted:
    full_frames = [...]                      # 整帧（roi=(0,0,1,1)）
    fallback = detector.verify_board_probable_paths(
        full_frames, required_frames=2,
        required_person_frames=settings.rv1106_verify_person_frames,
        board_score=..., board_person_score=...,
        ...
    )
    if fallback.accepted:
        verification = fallback
```

`verify_board_probable_paths`（`daughter_detector.py:404-429`）的放行分支：

```python
elif (face_frames == 0
      and person_frames >= required_people
      and board_score >= board_score_threshold
      and board_person_score >= board_person_score_threshold):
    accepted = True
    decision = "faceless_person"
```

问题在于：**`faceless_person` 只验证"画面里有人"，不验证"画面里是小孩"**。
奶奶背对镜头洗衣服时 `face_frames == 0`，人当然存在、分数也够，于是
`faceless_person` 放行 —— 整帧回退把板端的误报合法化了。

再加上 2026-11-04 为了适配 YOLO 的低置信度把
`RV1106_VERIFY_BOARD_SCORE` / `RV1106_VERIFY_BOARD_PERSON_SCORE` 下调过，
成年人更容易过线。

## 4. 建议改动（最小侵入）

**整帧回退路径不接受 `faceless_person`。** 该路径的存在意义是"救回板端框
指错位置的真实片段"，而不是"证明画面里有人"；如果孩子真的在画面里，
整帧抽帧通常能拿到儿童人脸（`child_face`），或 ROI 本来就不会失败。

```python
if not verification.accepted:
    ...
    if full_frames:
        fallback = detector.verify_board_probable_paths(...)
        # 只接受有儿童证据的结论
        if fallback.accepted and fallback.decision != "faceless_person":
            verification = fallback
        else:
            print(
                f"[INGEST] {job_dir.name}: full-frame fallback rejected "
                f"({fallback.decision})",
                flush=True,
            )
```

### 如果担心召回掉太多，可退一步

允许 faceless，但加两条硬约束（二选一或同时）：

1. `person_frames >= 4`（比原来的 3 更严）；
2. **画面里最大的人体框不得是成人尺寸**：在 `verify_board_probable_paths`
   里把 `max_person_score` 旁边再算一个 `max_person_height`
   （`person_boxes` 的归一化高度），要求 `max_person_height <= 0.45`。

> 依据：本地 640x360 实测，女儿站立约 0.37、走近 0.5+；成年人站立/贴近镜头
> 普遍 > 0.55。0.45 可以把"凑近镜头的大人"挡在外面。

### 建议加一个开关

```env
# true: 整帧回退允许 faceless_person (旧行为) ; false: 只接受儿童人脸证据
RV1106_VERIFY_FALLBACK_ALLOW_FACELESS=false
```

便于出问题时一键回滚，不必改代码。

## 5. 验收标准

改动上线后，用已知样本回放（片段可通过 `/api/moments/<id>/download` 取）：

| 片段 | 期望 | 理由 |
|---|---|---|
| 3784（07:05 只有爸爸） | **拒绝** | 无儿童证据 |
| 3791（10:04 只有奶奶） | **拒绝** | 无儿童证据 |
| 3786（07:34 女儿站茶几边） | **接受** | 有儿童（背影，可能走 faceless，需确认） |
| 3789（08:10 奶奶 + 女儿） | **接受** | 有儿童 |

如果 3786 因为"背影无脸"被拒，说明第二步的 `max_person_height` 兜底是必要的，
请启用 4.1 的退一步方案而不是直接放开。

## 6. 可观测性

每次回退都打一行日志（含 job key、decision、person_frames、board 分数），
便于事后统计"整帧回退救回多少、误放多少"：

```
[INGEST] <key>: roi-miss fallback decision=faceless_person person_frames=5 board=0.64/0.21 -> rejected
```

## 7. 板端配套（已完成，不需要 NAS 侧改）

- `relative_child` 加最小尺寸（`228060f`）
- 置信度按 `obj*cls` 乘积过滤、框宽/宽高比/贴边过滤（`bd0ddee`）
- `person_min_score=0.25`：低分框不建轨迹

板端修好后，误报源头会显著减少；NAS 侧这层加固是"最后一道闸"，
两层都做才能把"只有大人"的片段压到接近 0。

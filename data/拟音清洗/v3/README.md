# 拟音清洗 v3

`v3` 是已完成审核的全量 LLM 辅助分类与归一化版本，不参与任何数据切分。它保留候选、模型决策、二次拉丁消歧、物化结果和必要审核证据。此前的 v2 确定性中间结果已按项目清理策略移除。

目录：

- `input/candidates.jsonl`：23,954 条全量输入；每条都带分类所需的变体、还原目标、来源和候选特殊片段。其中包含 `08_补充数据_已处理` 的 578 条既有补充样本。
- `llm/annotations.jsonl`：大模型的严格结构化分类与归一化决策，支持断点续跑。
- `llm/failed.jsonl`：当前仍未成功的请求或输出校验失败记录；续跑成功后会自动移除陈旧失败项。
- `normalized/normalized_records.jsonl`：只从逐项决策物化出的标准化表层文本。
- `latin/`：拉丁候选二次消歧输入、模型策略和完整归一化审核表。完整拼音保留小写、中文缩写映射字典汉字、英文/外来表面统一大写。
- `final/真实数据_规则归一化.csv`：真实数据唯一的 IPA 前最终归一化 CSV；合成数据对应文件为 `synthetic/final/合成数据_规则归一化.csv`。
- `ipa/真实数据_分段IPA.csv`：真实数据的独立 IPA 衍生结果；合成数据对应文件为 `synthetic/ipa/合成数据_分段IPA.csv`。二者不回写 `final/`，并以 `ipa_segments_json` 保存逐段类型、读音路线和 IPA。
- `ipa/外来表面_发音待复核.csv`：无法按已接受的英文词、逐字母名、中文声母或完整拼音规则转换的片段。无口语读法的非拉丁残留、转义换行和格式字符在 IPA 层剔除，不进入该队列。
- `training/拟音还原_全量训练样本.json`：44,824 条可发音拟音训练三元组。`training/拟音还原_含恒等样本_训练样本.json` 在此前基础上加入 6,928 条 `identity` 样本；训练时必须保留并使用 `sample_type` 控制采样比例。
- `expansion/`：对未原始导入 v3 的 6,939 条独立对的审计。6,933 条无表层变异，作为恒等样本进入 combined training export；其余 6 条仍待首轮标注，尚未并入拟音训练集。
- `review/manual_review.csv`：首轮审核记录，仅作历史审核证据；最终文本以 `final/` 为准。

`normalized_phonetic_text` 不是 `canonical_text` 的替代品：它只展开模型已明确判定的数字、符号、Emoji、拼音或缩写表层读法。数字的读法以人工 `canonical_text` 的逐候选对齐证据和上下文唯一确定；即使答案中仍保留数字字形，归一化文本也写入对应汉字读法，例如 `94 -> 九四`，供后续 IPA 转换。拉丁二次消歧完成后，`final/` 中的文本才是 IPA 前唯一输入。原始 `variant_text` 与完整 `canonical_text` 始终保留。

`08_补充数据_已处理` 的 578 条记录在导入前已完成分类与归一化。调用器将其计为 `preprocessed_records` 并跳过 LLM；物化阶段直接使用保留在 `metadata_json.legacy_processing` 中的既有结果。

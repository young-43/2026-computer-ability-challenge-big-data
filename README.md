# 2026年计算机能力挑战赛大数据赛道思路

## 眼底图像青光眼分类：全图与局部融合

这是一个眼底图像二分类研究项目的匿名源码公开版。项目围绕一个问题展开：怎样结合整张眼底图像的整体信息与视盘附近的局部信息，提高阳性与阴性图像的排序质量？

最终参考方案使用全图分类器、局部分类器和一个视盘定位器。两个分类器按 **logit 等权融合**，再通过 sigmoid 得到连续分数。定位器只负责确定局部输入的位置，不参与最终类别分数融合。

本仓库提供推理实现、输入处理逻辑和研究过程说明。源码以普通目录展示，三个参考权重在 Releases 中分别下载。原始图像、逐样本标签、比赛提交文件与私人实验档案不公开。

## 方法

```text
输入眼底 JPEG
  ├─ 全图预处理 → 全图分类器 → global_logit
  └─ 视盘定位 → 固定比例局部裁剪 → 局部分类器 → local_logit
                           ↓
       sigmoid(0.5 × global_logit + 0.5 × local_logit)
```

- 输入保持原图宽高比，使用固定预处理与黑色填充。
- 局部输入以预测的视盘中心为基准，使用固定 30% 视野规则，分类器输入为 384 × 384。
- 裁剪几何定义见 `model/glaucoma/r27_geometry.py`；边界处理与回退行为以实现为准。
- 最终参考方案不使用年龄输入，也不使用额外阳性专家。
- 主要评价指标为连续分数的 AUROC，同时观察 AP。阈值 0.5 的分类结果仅作为辅助输出。

## 仓库结构

可以直接在 GitHub 中浏览以下目录，也可以下载或克隆仓库后运行。

| 路径 | 内容 |
|---|---|
| `code/predict.py` | 预测命令入口与运行精度设置 |
| `code/utils.py` | 图像与输入列表校验 |
| `code/submission_io.py` | 连续概率与类别输出 |
| `model/predict.py` | 全图、定位、局部与融合推理流程 |
| `model/glaucoma/` | 模型定义及预处理、定位和裁剪几何 |
| `model/config.json` | 固定参考配置与预期权重哈希 |
| `model/manifest.json` | 源码及外部权重的完整性要求 |
| `docs/research_notes.md` | 研究过程、实验解释与局限 |
| `tests/` | 不依赖图像、权重和 GPU 的基础检查 |
| `licenses/` | 相关第三方许可声明 |

## 使用条件

源码使用 Python、PyTorch、torchvision、NumPy 和 Pillow。依赖名称见 `requirements.txt`。历史运行环境为 Python 3.12。这里没有提供已经验证兼容的全平台安装环境。

从 [v1.0.0 Releases](https://github.com/young-43/2026-computer-ability-challenge-big-data/releases/tag/v1.0.0) 下载三份参考检查点，并分别放入：

```text
model/weights/global.pt
model/weights/local.pt
model/weights/locator.pt
```

代码会检查权重的哈希、结构及训练轮次信息；随机权重或其他同架构权重不能替代参考检查点。也可在仓库根目录运行 `python scripts/download_weights.py`，从固定版本下载并校验 SHA-256。

输入列表由使用者自行创建，CSV 必须包含 `Image_name` 列，其值为图像文件名。预测示例：

```bash
python code/predict.py --images-dir /path/to/images --metadata /path/to/images.csv --expected-count 100 --output /path/to/predictions.csv --device cuda
```

输出包含 `Image_name`、`Probability`、`Pred_Label`。已有输出文件不会被覆盖。示例路径及数量仅用于说明，仓库中没有真实样本或真实样本编号。

执行无需模型权重的检查：

```bash
python -B -m unittest discover -s tests -v
```

## 结果与适用范围

项目探索了裁剪范围、亮度处理、年龄辅助、排序损失、表征学习及轻量专家等方向。公开说明侧重方法和实验判断，不公开私人日志和逐样本结果。不同划分上的指标不能直接横向比较，重复开发过的验证集也不等于独立盲测。

推理代码保留参考实现的精度设置。不同 GPU、软件版本或 JPEG 解码环境可能造成数值差异；本公开版未执行真实模型前向或跨设备复测，不保证特定硬件的耗时或逐位一致性。

本项目仅用于研究和代码交流，不能作为临床诊断工具。

## 来源与许可

实现使用 PyTorch/torchvision 的模型接口，历史初始化来源涉及通用 ImageNet 预训练，相关第三方声明保存在 `licenses/`。这些声明仅适用于对应第三方材料。

项目原创代码和本项目发布的参考权重采用 [MIT License](LICENSE)。第三方材料仍适用各自许可，见 `licenses/`。本仓库不授予原始数据的使用权。

## 权重版本与验证

| 文件 | 模型作用 | 大小（约） |
|---|---|---:|
| global.pt | EfficientNet-B2 全图分类 | 31.2 MB |
| local.pt | ConvNeXt-Tiny 局部分类 | 111.3 MB |
| locator.pt | 视盘中心定位 | 0.9 MB |

公开权重清除了训练配置中的服务器路径并重新封装；864 个张量存储块的 SHA-256 均与私人原材料逐块一致。文件哈希因脱敏与封装而改变，需使用本仓库的配置与校验文件。没有使用医学外部预训练权重。

本次发布没有执行真实图像推理或 PyTorch 加载复测。已完成 ZIP 完整性、检查点结构的受限解析、张量存储块一致性和源码基础检查；这些检查不能代替端到端模型复现。

## 实验结果如何理解

以下是同一份反复开发的验证集上的历史对照，非独立盲测，也不是平台分数：

| 方案 | AUROC | AP |
|---|---:|---:|
| 全图与局部固定等权融合 | 98.9576% | 0.8711 |
| 加入固定 10% 轻量专家 | 99.0161% | 0.8647 |

专家融合的 AUROC 差异置信区间跨零，AP 同时下降，因此数值增加不足以支持替换最终参考方案。不同研究使用过不同划分，不能把其他集合上的专家单模型指标直接放到这张表中比较。

数据未公开，因此这里的历史结果不能由仓库单独重新计算。研究过程和评价局限见 [研究记录](docs/research_notes.md)。

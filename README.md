# Art Center Client

消费级显卡可用的画师参考图集预处理客户端。它会遍历本地图片目录，为图片生成 SigLIP2 向量、自动标注参考类别，并在识别到“动作/姿态”类图片时提取人体动作参数，写入 PostgreSQL，作为后续自然语言检索和 Blender 动作应用的基础数据。

本项目是 Art Center 工作流的第一阶段：**预处理**。

## 项目目标

画师通常会长期收集大量参考图，但真正需要某类服装、构图、配色、发型或动作时，手动翻找效率很低。Art Center 的目标是把这些私人参考图集转化为可检索、可复用、可回溯的专属参考库：

- **低门槛**：经测试 RTX 3060 级别消费显卡可用。
- **高定制**：完全基于画师自己的图集，不依赖通用图库。
- **高可控**：不是直接生成图片，而是检索真实参考图，并将参考动作转为可在 Blender 中微调的骨骼动作。
- **可回溯**：输出保留原始图片路径，画师可以随时回到原图确认来源。

## 完整工作流

Art Center 分为三个阶段：

1. **预处理（本项目）**
   - 修改 `main.bat` 中的图片文件夹路径。
   - 启动脚本后，程序会遍历 `.png`、`.jpg`、`.jpeg` 图片。
   - 使用 SigLIP2 为图片生成向量。
   - 自动为图片标注参考类别。
   - 对类别为动作的图片提取人体动作参数，生成动作库数据。
   - 将图片路径、向量、类别、动作 JSON 写入 PostgreSQL。

2. **获取参考**
   - 服务端项目：[art_center-server](https://github.com/nar-oah/art_center-server)
   - 画师输入抽象需求。
   - Gemini 将需求转换为适合 SigLIP2 的英文短描述。
   - 服务端将描述转为向量，并在数据库中匹配参考图。
   - 返回匹配到的图片路径；如果匹配的是动作类别，还会返回动作 JSON。

3. **动作处理**
   - Blender 插件项目：[art_center-plugin](https://github.com/nar-oah/art_center-plugin)
   - 在 Blender 中启动插件。
   - 准备 A-Pose、带骨骼的模型，骨骼名称最好与插件约定一致。
   - 选择模型骨架并上传上一步得到的动作 JSON。
   - 插件会将动作应用到模型骨骼上，画师仍可继续调整摄像机角度、构图和姿态细节。

## 功能

- 递归扫描图片目录。
- 使用 `google/siglip2-base-patch16-naflex` 提取图片向量。
- 使用 SigLIP2 图文匹配结果为图片标注类别：
  - `background element`
  - `layout or composition`
  - `color pairing`
  - `clothing combination`
  - `clothing silhouette`
  - `clothing style or material`
  - `hairstyle or hair ornament`
  - `facial features or makeup`
  - `pose or body gesture`
- 对 `pose or body gesture` 类图片运行动作提取。
- 使用 YOLO 检测人体区域。
- 使用 SMPLest-X 风格模型提取身体和手部动作参数。
- 通过 `ON CONFLICT (path) DO UPDATE` 方式写入或更新数据库记录。

## 项目结构

```text
.
├── main.py              # 主入口：扫描图片、提取向量、提取动作、写入数据库
├── main.bat             # Windows 一键启动脚本
├── analyzer.py          # SigLIP2 图片向量和类别分析
├── db.py                # PostgreSQL 连接和 upsert
├── requirements.txt     # Python 依赖
└── pose/
    ├── inference.py     # 动作提取流程
    ├── module.py        # 模型模块定义
    ├── smplx.py         # SMPLest-X 包装
    └── models/          # 本地模型权重目录，不应提交
```

## 环境要求

- Python 3.12+
- NVIDIA 显卡和可用 CUDA 环境
- PostgreSQL
- 已创建用于保存结果的 `art_center` 表
- 首次运行 SigLIP2 时需要能从 Hugging Face 下载模型，或提前准备好本地缓存

> `psycopg.connect()` 会读取标准 PostgreSQL 环境变量，例如 `PGHOST`、`PGPORT`、`PGDATABASE`、`PGUSER`、`PGPASSWORD`。

## 安装

创建并激活虚拟环境：

```bash
python -m venv venv
source venv/bin/activate
```

Windows：

```bat
python -m venv venv
venv\Scripts\activate.bat
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 模型权重

动作提取需要以下本地权重文件：

```text
pose/models/yolo26n.pt
pose/models/smplest_x_h.pth.tar
```

`pose/models/` 是本地目录，不应提交到 Git。请根据模型来源自行下载并放到对应路径。

SigLIP2 默认使用：

```text
google/siglip2-base-patch16-naflex
```

该模型由 `transformers` 在首次运行时自动加载。

## 数据库

程序会写入 `art_center` 表，字段由 `db.py` 使用：

| 字段 | 含义 |
| --- | --- |
| `path` | 原始图片路径，应具有唯一约束 |
| `vector` | SigLIP2 图片向量 |
| `categories` | 图片类别列表 |
| `pose` | 动作 JSON；非动作类图片为 `NULL` |

写入逻辑：

```sql
INSERT INTO art_center (path, vector, categories, pose)
VALUES (...)
ON CONFLICT (path) DO UPDATE
SET vector = EXCLUDED.vector,
    categories = EXCLUDED.categories,
    pose = EXCLUDED.pose;
```

数据库表结构需要与服务端检索逻辑保持一致，通常由服务端项目或部署脚本统一准备。

## 使用方式

### Windows 批处理运行

打开 `main.bat`，把最后的图片目录改成你的参考图集目录：

```bat
python main.py /d/Resources/
```

例如：

```bat
python main.py D:\ArtReferences
```

然后双击运行 `main.bat`。

### 命令行运行

```bash
python main.py /path/to/images
```

程序会显示两个进度条：

- `获取特征`：提取图片向量和类别。
- `获取动作`：将数据写入数据库，并对动作类图片提取动作 JSON。

## 输出数据

每张图片会写入一条数据库记录：

- `path`：图片绝对路径或传入目录下扫描得到的路径。
- `vector`：SigLIP2 图片嵌入向量。
- `categories`：一个或多个参考类别。
- `pose`：动作数据，结构包含：

```json
{
  "body_root_pose": [],
  "body_pose": [],
  "lhand_pose": [],
  "rhand_pose": []
}
```

如果图片未被分类为 `pose or body gesture`，或没有检测到人体，则 `pose` 可能为空。

## 开发检查

当前项目没有完整自动化测试。可以使用下面的命令做轻量语法检查：

```bash
python -m compileall main.py analyzer.py db.py pose
```

## 常见问题

### 显存不够怎么办？

先确认没有其他程序占用 GPU。项目会在图片向量提取后释放 SigLIP2，再加载动作模型，以降低显存压力。如果图集很大，建议先用较小目录试跑，确认环境和数据库配置无误后再处理完整图集。

### 为什么动作数据为空？

常见原因：

- 图片未被分类为 `pose or body gesture`。
- YOLO 没有检测到人体。
- 人体区域过小、遮挡严重或构图不适合动作提取。
- `pose/models/` 中权重文件缺失或路径不正确。

### 数据库连接失败怎么办？

确认 PostgreSQL 环境变量已经设置：

```bash
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=your_database
export PGUSER=your_user
export PGPASSWORD=your_password
```

Windows 可使用系统环境变量或在启动终端中设置对应变量。

### 可以直接用 AI 生成参考图吗？

这个项目的重点不是替代画师判断，也不是直接生成图片，而是提升已有参考图集的使用率。它会帮助画师从自己的图集中找出符合需求的图片，并把动作转换到可编辑的 3D 模型上，使参考过程仍然可控、可微调、可回溯。

## 相关项目

- 服务端检索：[art_center-server](https://github.com/nar-oah/art_center-server)
- Blender 动作插件：[art_center-plugin](https://github.com/nar-oah/art_center-plugin)


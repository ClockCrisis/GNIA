一年前的代码, 也放到github上吧, 如需复现, 请按照以下步骤操作
复现不了,不要联系我, 因为我也不记得了, 只能求神拜佛了
---小钟 

## 项目结构

```
├── cresci_2015/
│   ├── raw_data/              # 原始数据
│   ├── processed_data/        # 处理后的数据
│   ├── model/                 # 模型存储目录
│   ├── checkpoint/            # 检查点存储目录
│   ├── utils.py               # 工具函数
│   ├── model.py               # 模型定义
│   ├── preprocess_1.py       # 预处理脚本1
│   ├── preprocess_2.py       # 预处理脚本2
│   ├── preprocess_3.py       # 预处理脚本3
│   ├── train.py              # 训练替代模型/GCN/HGT/SimpleHGN/RGCN
│   ├── Dataset.py            # 数据集类
│   ├── preprocess.py          # 数据集预处理
│   ├── dataset_tool.py        # 数据集工具
│   ├── cat_decoder.py        # 训练类别属性解码器
│   ├── num_decoder.py        # 训练数值属性解码器
│   ├── gnia.py                # GNIA攻击模型
│   ├── run_gnia.py           # 训练攻击模型
│   ├── layer.py              # 网络层定义
│   ├── test_GCN.py           # 在GCN模型上测试攻击效果
│   ├── test_HGT.py           # 在HGT模型上测试攻击效果
│   ├── test_SimpleHGN.py      # 在SimpleHGN模型上测试攻击效果
│   └── test_RGCN.py          # 在RGCN模型上测试攻击效果
├── twibot_22/
│   ├── raw_data/              # 原始数据
│   ├── processed_data/        # 处理后的数据
│   ├── model/                 # 模型存储目录
│   ├── checkpoint/            # 检查点存储目录
│   ├── utils.py               # 工具函数
│   ├── model.py               # 模型定义
│   ├── preprocess_1.py       # 预处理脚本1
│   ├── preprocess_2.py       # 预处理脚本2
│   ├── preprocess_3.py       # 预处理脚本3
│   ├── train.py              # 训练替代模型/GCN/HGT/SimpleHGN/RGCN
│   ├── Dataset.py            # 数据集类
│   ├── preprocess.py          # 数据集预处理
│   ├── sub-graph.py          # 子图提取
│   ├── dataset_spilt.py       # 选择子图
│   ├── dataset_tool.py        # 划分子图数据集
│   ├── cat_decoder.py        # 训练类别属性解码器
│   ├── num_decoder.py        # 训练数值属性解码器
│   ├── gnia.py                # GNIA攻击模型
│   ├── run_gnia.py           # 训练攻击模型
│   ├── layer.py              # 网络层定义
│   ├── test_GCN.py           # 在GCN模型上测试攻击效果
│   ├── test_HGT.py           # 在HGT模型上测试攻击效果
│   ├── test_SimpleHGN.py      # 在SimpleHGN模型上测试攻击效果
│   └── test_RGCN.py          # 在RGCN模型上测试攻击效果
└── readme.md
```

## 实现细节

由于部分相关数据缺失，用户数值属性和用户类别属性与原始设计有所不同。

### 1. 数值属性

- **原始设计** (dim=6)：
  - followers + followings + favorites + statuses + active_days + screen_name_length

- **cresci-2015 / twibot-22** (dim=5)：
  - followers + followings + statuses + active_days + screen_name_length

### 2. 类别属性

- **原始设计** (dim=11)：
  - protected + verified + default_profile_image + geo_enabled + contributors_enabled + is_translator + is_translation_enabled + profile_background_image + profile_user_background_image + has_extended_profile + default_profile

- **cresci-2015** (dim=1)：
  - default_profile_image

- **twibot-22** (dim=3)：
  - protected + verified + default_profile_image

## 实验复现步骤

### 1. 进入数据集目录

- cresci-2015 数据集：`cd cresci_2015/`
- twibot-22 数据集：`cd twibot_22/`

### 2. 预处理数据集

```bash
python preprocess.py
```

### 3. 训练替代模型 / GCN / HGT / SimpleHGN / RGCN

```bash
python train.py
```

### 4. 训练数值属性解码器

```bash
python num_decoder.py
```

### 5. 训练类别属性解码器

```bash
python cat_decoder.py
```

### 6. (仅 twibot-22 数据集) 选择并划分子图

```bash
python dataset_spilt.py
python dataset_tool.py
```

### 7. 训练攻击模型

```bash
python run_gnia.py
```

### 8. 测试攻击模型

```bash
python test_GCN.py
python test_HGT.py
python test_SimpleHGN.py
python test_RGCN.py
```

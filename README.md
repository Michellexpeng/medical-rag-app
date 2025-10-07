# 医学RAG应用使用指南

## 📋 项目概述

本项目是一个专门处理医学教材PDF的RAG（Retrieval-Augmented Generation）系统，基于RAG-Anything框架构建，针对医学文档的特点进行了优化。

## 🏗️ 项目结构

```
medical-rag-app/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python依赖包列表
├── .env.example                       # 环境变量配置模板
├── data/                              # 数据目录
│   ├── raw/                          # 原始医学PDF文档
│   │   ├── CT and MRI of the Whole Body...pdf
│   │   ├── Diagnostic Imaging_ Abdomen...pdf
│   │   └── ... (其他医学教材PDF)
│   └── indexes/                      # 索引文件存储
├── logs/                             # 日志文件目录
├── rag_storage/                      # RAG系统存储目录
├── scripts/                          # 处理脚本目录
│   ├── medical_rag_processor.py      # 单文档处理器
│   ├── batch_medical_processor.py    # 批量处理器
│   ├── medical_rag_processor_lite.py # 轻量级处理器（避免超时）
│   └── medical_rag_query.py          # 英文查询工具
└── src/
    └── medical_rag/                  # 源代码模块
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 1. 创建并激活Python虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或 venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装RAG-Anything (如果还没安装)
cd ../RAG-Anything
pip install -e .
cd ../medical-rag-app
```

### 2. 配置环境变量

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑.env文件，填入您的OpenAI API密钥
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. 运行处理器

#### 单文档处理

```bash
# 处理单个医学PDF
python scripts/medical_rag_processor.py --file "data/raw/CT and MRI of the Whole Body, 2-Volume Set, 6e, Volume I ( PDFDrive ).pdf"

# 交互式查询模式
python scripts/medical_rag_processor.py --file "data/raw/Liver imaging _ MRI with CT correlation ( PDFDrive ).pdf" --interactive

# 指定输出目录
python scripts/medical_rag_processor.py --file "data/raw/Diagnostic Imaging_ Abdomen_ Published by Amirsys® ( PDFDrive ).pdf" --output ./custom_output
```

#### 批量处理

```bash
# 处理所有医学PDF文档
python scripts/batch_medical_processor.py --all

# 处理指定文档
python scripts/batch_medical_processor.py --files "CT and MRI of the Whole Body, 2-Volume Set, 6e, Volume I ( PDFDrive ).pdf" "Liver imaging _ MRI with CT correlation ( PDFDrive ).pdf"

# 设置并行处理数量
python scripts/batch_medical_processor.py --all --max-workers 3
```

#### 查询已处理数据（推荐）

```bash
# 使用英文查询已处理的医学数据（无需重新处理）
python scripts/medical_rag_query.py --working-dir ./rag_storage

# 交互式英文查询
python scripts/medical_rag_query.py --working-dir ./rag_storage --interactive

# 单个英文查询
python scripts/medical_rag_query.py --working-dir ./rag_storage --query "What are the main imaging techniques discussed?"
```

## 📚 功能特性

### 🏥 医学文档处理优化

- **多模态内容支持**: 处理医学图像、表格、公式
- **专业解析器**: 使用MinerU解析器，对医学PDF支持更好
- **智能解析**: 自动选择最佳解析方法

### 🔍 医学专业查询

- **基础医学概念查询**: 影像学检查方法、解剖结构、疾病状态
- **图像分析查询**: 医学影像描述、征象讨论
- **多模态查询**: 结合临床数据表格的综合分析
- **英文查询支持**: 专门的英文查询工具，优化的token使用
- **VLM增强查询**: 自动分析医学图像并结合文本内容

### ⚡ 批量处理能力

- **并行处理**: 支持多文档同时处理
- **进度跟踪**: 实时显示处理进度
- **错误恢复**: 单个文件失败不影响其他文件

### 🛠️ 高级功能

- **交互式查询**: 实时问答模式
- **详细日志**: 完整的处理日志记录
- **灵活配置**: 多种参数可调

## 📖 使用示例

### 单文档处理示例

```python
# 基础用法
python scripts/medical_rag_processor.py \
  --file "data/raw/CT and MRI of the Whole Body, 2-Volume Set, 6e, Volume I ( PDFDrive ).pdf" \
  --output "./output" \
  --parser mineru
```

### 批量处理示例

```python
# 处理所有PDF
python scripts/batch_medical_processor.py \
  --all \
  --output "./batch_output" \
  --max-workers 2
```

### 英文查询示例（推荐）

使用新的英文查询工具：

```bash
# 运行预设的英文医学查询（已优化，避免API限制）
python scripts/medical_rag_query.py --working-dir ./rag_storage
```

查询示例（自动运行）：
- "What is the main content of this medical textbook?"
- "What imaging techniques are discussed in this document?"
- "What are the differences between CT and MRI imaging?"
- "What clinical scenarios are presented in this textbook?"

### 交互式查询示例

```bash
# 中文交互模式
python scripts/medical_rag_processor.py --file "文件路径" --interactive

# 英文交互模式
python scripts/medical_rag_query.py --working-dir ./rag_storage --interactive
```

中文查询示例：
```
❓ 请输入您的医学问题: CT扫描的基本原理是什么？
❓ 请输入您的医学问题: 肝脏影像学检查有哪些方法？  
❓ 请输入您的医学问题: MRI和CT在腹部成像中的区别？
❓ 请输入您的医学问题: 胰腺炎的影像学表现有哪些？
```

英文查询示例：
```
❓ Enter your medical question: What are the main radiological findings in pneumonia?
❓ Enter your medical question: How do you differentiate between benign and malignant lesions on CT?
❓ Enter your medical question: What are the contraindications for MRI contrast agents?
```

## 🔧 配置说明

### 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API密钥 | 必填 |
| `OPENAI_BASE_URL` | API基础URL | https://api.openai.com/v1 |
| `LOG_DIR` | 日志目录 | ./logs |
| `RAG_WORKING_DIR` | RAG工作目录 | ./rag_storage |
| `MAX_WORKERS` | 最大并行数 | 2 |

### 处理器参数

#### medical_rag_processor.py

- `--file`: 医学PDF文档路径 (必填)
- `--output`: 输出目录 (默认: ./output)
- `--working-dir`: RAG存储目录 (默认: ./rag_storage)
- `--parser`: 解析器类型 (默认: mineru)
- `--interactive`: 启用交互模式
- `--verbose`: 启用详细日志

#### batch_medical_processor.py

- `--all`: 处理data/raw/下所有PDF
- `--files`: 指定处理的文件列表
- `--max-workers`: 最大并行数 (默认: 2)
- `--output`: 批量输出目录 (默认: ./batch_output)
- `--data-dir`: 数据目录路径 (默认: ./data)

#### medical_rag_query.py（推荐）

- `--working-dir`: RAG存储目录路径 (必填)
- `--interactive`: 启用交互模式
- `--query`: 单次查询文本
- 无参数: 运行预设的医学查询示例

#### medical_rag_processor_lite.py

- 与medical_rag_processor.py相同参数
- 优化了大文档处理，避免超时问题
- 使用分块嵌入策略

## 📝 日志和输出

### 日志文件

- 单文档处理: `logs/medical_rag_processor.log`
- 批量处理: 控制台输出 + 详细统计

### 输出结构

```
output/
├── document_name/           # 每个文档的输出目录
│   ├── parsed_content/     # 解析后的内容
│   ├── images/             # 提取的图像
│   ├── tables/             # 提取的表格
│   └── metadata.json       # 文档元数据
└── ...
```

## ✅ 成功案例

### 已验证处理的医学教材

- ✅ **CT and MRI of the Whole Body** - 完整处理，1434个文本块
- 📊 实体数量: 7,671个医学实体
- 📊 关系数量: 21,425个医学关系
- 🔍 支持英文和中文查询
- 🖼️ 支持VLM增强的图像分析

### 成功查询示例

**基础医学查询**：
- "What is the main content of this medical textbook?" ✅
- "What imaging techniques are discussed?" ✅
- "What anatomical structures are covered?" ✅

**技术查询**：
- "What are the differences between CT and MRI imaging?" ✅
- "What are the main radiological findings discussed?" ✅

**临床应用查询**：
- "What clinical scenarios are presented?" ✅
- "What diagnostic criteria are mentioned?" ✅

## ⚠️ 注意事项

1. **推荐使用**: 使用`medical_rag_query.py`查询已处理数据，避免重复处理
2. **API限制**: 已优化token使用，设置查询间隔避免速率限制
3. **内存使用**: 大文档处理可能需要较多内存，可使用lite版本
4. **并行处理**: 建议并行数不超过3，避免API限制
5. **文件路径**: 确保PDF文件路径正确，支持中文路径
6. **网络连接**: 需要稳定的网络连接访问OpenAI API
7. **嵌入维度**: 确保处理和查询使用相同的嵌入模型维度

## 🆘 故障排除

### 常见问题

1. **API密钥错误**
   ```
   ❌ 未提供OpenAI API密钥
   ```
   解决: 检查.env文件中的OPENAI_API_KEY配置

2. **文件不存在**
   ```
   ❌ 文件不存在: data/raw/xxx.pdf
   ```
   解决: 检查文件路径和文件名是否正确

3. **嵌入维度不匹配**
   ```
   ❌ Embedding dim mismatch, expected: 3072, but loaded: 1536
   ```
   解决: 确保查询工具使用与处理时相同的嵌入模型

4. **API速率限制**
   ```
   ❌ Rate limit reached for gpt-4o-mini
   ```
   解决: 使用优化的查询工具，已设置自动延迟

5. **QueryParam参数错误**
   ```
   ❌ QueryParam.__init__() got an unexpected keyword argument
   ```
   解决: 使用正确的LightRAG QueryParam参数

6. **解析失败**
   ```
   ❌ 处理医学文档时出错: ...
   ```
   解决: 尝试使用lite版本处理器或检查PDF文件完整性

7. **内存不足**
   ```
   内存错误或处理缓慢
   ```
   解决: 使用lite版本处理器或减少并行数量

### 获得帮助

如果遇到问题，请：

1. 检查日志文件获取详细错误信息
2. 确认环境配置正确
3. 尝试使用--verbose参数获取更多调试信息
4. 从单个小文件开始测试

## 🔄 版本更新

当RAG-Anything框架更新时：

```bash
cd ../RAG-Anything
git pull
pip install -e .
cd ../medical-rag-app
```

## 📈 性能优化建议

1. **优先使用查询工具**: 用`medical_rag_query.py`查询已处理数据，避免重复处理
2. **选择合适的处理器**: 
   - 普通文档: `medical_rag_processor.py`
   - 大文档: `medical_rag_processor_lite.py`
   - 批量处理: `batch_medical_processor.py`
3. **API使用优化**:
   - 使用较小的模型（gpt-4o-mini）降低成本
   - 设置合理的token限制避免速率限制
   - 使用local模式而非hybrid模式减少复杂度
4. **并行设置**: 根据系统配置调整max-workers
5. **缓存利用**: RAG系统会自动缓存处理结果
6. **分批处理**: 对于大量文档，考虑分批处理

## 🎯 推荐工作流程

1. **首次处理**: 使用处理器处理医学PDF
   ```bash
   python scripts/medical_rag_processor_lite.py --file "path/to/medical.pdf"
   ```

2. **后续查询**: 使用查询工具进行英文查询
   ```bash
   python scripts/medical_rag_query.py --working-dir ./rag_storage
   ```

3. **交互使用**: 根据需要使用交互模式
   ```bash
   python scripts/medical_rag_query.py --working-dir ./rag_storage --interactive
   ```

---

祝您使用愉快！如有问题，请查看日志文件或联系技术支持。

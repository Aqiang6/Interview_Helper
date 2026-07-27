# AI Agent Interviewer - 智能模拟面试系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green)
![Vue](https://img.shields.io/badge/Vue-3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

一个现代化的 AI 模拟面试系统，支持简历智能解析、针对性技术面试、语义缓存优化和可视化评估报告。专为求职者、面试官和技术面试准备者设计。

## ✨ 核心功能

### 📄 智能简历解析
- **多格式支持**: PDF 文件上传、纯文本粘贴
- **自动信息提取**: 姓名、学历、工作年限、技术栈、项目经历
- **OCR 增强**: 自动检测扫描件 PDF，调用 Tesseract OCR 识别中文
- **乱码处理**: 智能识别并降级处理编码错误

### 🤖 AI 模拟面试
- **简历驱动**: 基于简历内容自动生成面试问题
- **技术八股文**: 针对 Java、Spring、数据库、分布式等核心技术提问
- **项目拷打**: 深入询问项目细节、技术选型和架构设计
- **流式响应**: SSE 实时展示面试官提问过程

### 📊 面试评估报告
- **综合评分**: 0-100 分综合评价
- **技能分析**: 各项技能的掌握程度评估
- **改进建议**: 针对弱项提供学习建议
- **对话回顾**: 完整的面试对话记录

### ⚡ 高级特性
- **语义缓存引擎**: 智能缓存相似对话，降低 API 调用成本
- **多 LLM 支持**: OpenAI、DeepSeek、通义千问、智谱 GLM 等
- **可观测性**: OpenTelemetry 集成，监控性能指标
- **加密存储**: 用户 API Key 在浏览器端加密存储

## 🚀 快速开始

### 环境要求
- Python 3.10+
- LLM API Key（OpenAI / DeepSeek / 通义千问等）
- Windows / macOS / Linux

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/ai-agent-interviewer.git
cd ai-agent-interviewer

# 2. 安装依赖
pip install -e .

# 3. 配置环境变量（可选 OCR）
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

### 启动服务

```bash
# 方式一：直接启动
python -m ai_interviewer.app

# 方式二：Windows 快速启动
start_app.bat
```

访问 [http://localhost:8000](http://localhost:8000) 开始使用。

## 📖 使用流程

1. **步骤 1 - 模型设置**
   - 选择预设模型（OpenAI、DeepSeek、通义千问、智谱 GLM）
   - 或手动填写 API Key、Base URL、模型名称
   - 点击“测试连接”验证配置

2. **步骤 2 - 上传简历**
   - **粘贴文本**: 直接输入或粘贴简历内容
   - **上传 PDF**: 拖拽或选择 PDF 文件（支持扫描件）
   - 点击“解析简历”，系统自动提取个人信息和技术栈

3. **步骤 3 - 模拟面试**
   - AI 面试官开始提问，基于简历内容
   - 在输入框中回答每个问题，按 Enter 发送
   - 面试结束后点击“结束面试并查看报告”

4. **步骤 4 - 面试报告**
   - 查看综合评分（0-100）
   - 分析技能掌握情况
   - 获取改进建议
   - 回顾完整面试对话

## 📁 项目结构

```
ai-agent-interviewer/
├── ai_interviewer/              # Python 后端
│   ├── cache_engine/            # 语义缓存引擎
│   │   ├── semantic_cache.py    # 语义缓存核心
│   │   ├── embedding.py         # 向量嵌入
│   │   ├── summarizer.py        # 对话摘要压缩
│   │   ├── metrics.py           # 缓存指标监控
│   │   └── benchmark.py         # 性能基准测试
│   ├── __init__.py
│   ├── app.py                   # FastAPI 应用入口
│   ├── config.py                # 配置管理
│   ├── interview_agent.py       # 面试对话 Agent
│   ├── models.py                # 数据模型
│   └── resume_parser.py         # 简历解析器（PDF/文本）
├── frontend/                    # 前端界面
│   └── index.html               # 单页 Vue 3 应用
├── .env.example                 # 环境变量模板
├── .gitignore                   # Git 忽略文件
├── pyproject.toml               # 项目依赖配置
├── start_app.bat                # Windows 启动脚本
├── run_benchmark.bat            # 缓存基准测试
└── README.md                    # 本文件
```

## 🔧 技术架构

### 后端架构
- **Web 框架**: FastAPI - 高性能异步 API
- **PDF 解析**: PyMuPDF (fitz) / pdfplumber / PyPDF2
- **OCR 引擎**: Tesseract 5.4 + pytesseract（扫描件支持）
- **AI 缓存**: Sentence-Transformers + 余弦相似度检索
- **可观测性**: OpenTelemetry（日志、指标、追踪）
- **加密**: cryptography.Fernet（API Key 加密）

### 前端架构
- **框架**: Vue 3 + Composition API
- **UI 组件库**: Element Plus
- **状态管理**: Vue 响应式数据
- **CSS**: CSS 自定义属性设计系统
- **API 通信**: Fetch API + SSE

### 缓存系统
- **向量嵌入**: sentence-transformers/all-MiniLM-L6-v2
- **相似度阈值**: 0.92（可配置）
- **TTL**: 3600 秒（可配置）
- **最大容量**: 10000 条缓存记录
- **压缩策略**: Token 阈值 + 摘要生成

## ⚙️ 配置说明

### 环境变量
```ini
# LLM 配置
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 备用模型（主模型失败时使用）
FALLBACK_API_KEY=sk-your-fallback-key
FALLBACK_BASE_URL=https://api.openai.com/v1
FALLBACK_MODEL=gpt-3.5-turbo

# 语义缓存配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CACHE_SIMILARITY_THRESHOLD=0.92
CACHE_TTL_SECONDS=3600
CACHE_MAX_SIZE=10000

# 摘要压缩
SUMMARY_TOKEN_THRESHOLD=4000
SUMMARY_TARGET_RATIO=0.4

# 服务配置
HOST=0.0.0.0
PORT=8000
```

### OCR 安装（扫描件 PDF）
如果你的 PDF 是扫描件（图片格式），需要安装 Tesseract：

**Windows**:
1. 下载 [Tesseract 5.4](https://github.com/UB-Mannheim/tesseract/wiki)
2. 安装到 `D:\Tesseract-OCR` 或 `C:\Program Files\Tesseract-OCR`
3. 安装时勾选中文语言包（chi_sim）

## 📈 性能优化

### 语义缓存优势
- **API 调用减少**: 相似问题自动命中缓存
- **响应时间缩短**: 缓存命中时无需调用 LLM
- **成本降低**: 减少重复的 Token 消耗

### 基准测试
```bash
# 运行缓存基准测试
run_benchmark.bat
```

测试指标包括：
- 缓存命中率
- 平均响应时间
- Token 使用量统计
- 相似度分布分析

## 🔍 故障排除

### PDF 解析问题
1. **扫描件识别失败** → 安装 Tesseract OCR
2. **中文乱码** → 系统会自动降级到 OCR 引擎
3. **文件过大** → 建议压缩 PDF 或分页上传

### API 连接问题
1. **测试连接失败** → 检查 API Key 和 Base URL
2. **响应超时** → 尝试使用备用模型
3. **流式中断** → 检查网络连接稳定性

### 界面问题
1. **聊天框太小** → 已优化为自适应布局，刷新页面
2. **步骤导航消失** → 步骤 2-4 显示为紧凑圆点导航
3. **样式错乱** → 清除浏览器缓存

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 Web 框架
- [Vue 3](https://vuejs.org/) - 渐进式 JavaScript 框架
- [Element Plus](https://element-plus.org/) - Vue 3 UI 组件库
- [Sentence Transformers](https://www.sbert.net/) - 向量嵌入模型
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR 引擎

---

**立即体验 AI 模拟面试**: [http://localhost:8000](http://localhost:8000)

如有问题或建议，请在 GitHub 仓库提交 Issue。
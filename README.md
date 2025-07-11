# LLM-PaperGen

LLM-PaperGen 是一个让大语言模型自动生成学术论文的流程。主要依赖 **大模型写论文.py** 脚本，组织数据、生成代码，并在运行失败时通过 LLM 动态修正代码。

## 主要文件

- **大模型写论文.py** — 完整的论文生成流程
- **prompts.json** — 向 LLM 发送的每一步 prompt
- **字段结构.json** — 字段描述信息
- **cfps2022.csv** — CFPS 2022 年数据集
- **cfps2022codebook.csv** — 数据字典（字段名及含义）
- **summ_detail_stats.csv** — 预先计算的统计量

## 支持的模型

脚本目前已支持 **Grok**、**DeepSeek-R1 0528**、**OpenAI(o3)**、**Gemini 2.5 Pro** 与 **Claude‑4 Opus** 五种模型，并依次对每个模型执行全部步骤。运行前需要设置相应的 API Key：

- `GROK_API_KEY`
- `DEEPSEEK_API_KEY`
- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `ANTHROPIC_API_KEY`

## 运行流程

1. **主题与字段预选**（STEP 1）
   - 用户可输入研究主题，或留空由 LLM 自行发挥
   - 根据 codebook 抽样字段供模型选择
2. **研究方案与字段筛选**（STEP 2、3）
   - LLM 制定研究计划并确定最终字段
3. **生成并运行代码**（STEP 5、7）
   - LLM 生成绘图和回归脚本，脚本会在出错时调用 LLM 尝试两次自动修复
   - 产出 `plot_*.png` 与 `result_*.txt`
4. **结果分析与摘要**（STEP 9、10）
   - LLM 根据回归结果给出分析并生成论文摘要
5. **整合 Markdown 文档**（STEP 11）
   - 汇总所有内容生成 `paper_*.md` 与 `paper_*.pdf`

执行完毕后，会在当前目录下生成一套包含图片、结果文本、Markdown 论文和 PDF 成品的文件。

## 使用方法

确保已安装 Python3 以及 `pandas`、`matplotlib`、`statsmodels` 等依赖，在数据文件所在目录运行：

```bash
python 大模型写论文.py
```

脚本会依次使用所有模型生成论文版本。

## 新增论文样本

我们已经更新了使用 `Gemini 2.5 Pro` 和 `DeepSeek-R1 0528` 新模型的分析，并新增了对 `Grok` 模型的支持。最新的论文样本（`paper_20250711_*.pdf`）已上传，欢迎查阅。

## 许可证

本项目采用 MIT 协议发布。
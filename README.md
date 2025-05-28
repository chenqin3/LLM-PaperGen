# LLM-PaperGen

LLM-PaperGen 是一个让大语言模型自动生成学术论文的流程。主要依赖 **大模型写论文.py** 脚本，组织数据、生成代码，并在运行失败时通过 LLM 动态修正代码。

## 主要文件

- **大模型写论文.py** — 完整的论文生成流程
- **prompts.json** — 向 LLM 发送的每一步 prompt，其中 `step12_fix_code` 用于自动修复脚本错误
- **字段结构.json** — 字段描述信息
- **cfps2022.csv** — CFPS 2022 年数据集
- **cfps2022codebook.csv** — 数据字典（字段名及含义）
- **summ_detail_stats.csv** — 预先计算的统计量

## 支持的模型

脚本内置 DeepSeek、OpenAI(o3)、Gemini 2.5 与 Claude‑4 Opus 四种模型，并依次对每个模型执行全部步骤。运行前需要设置：

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
   - LLM 生成绘图和回归脚本
   - 若脚本运行出错，会将错误代码和回溯信息连同原始指令传入 `step12_fix_code`，
     由模型返回修正后的代码并最多尝试两次
   - 产出 `plot_*.png` 与 `result_*.txt`
4. **结果分析与摘要**（STEP 9、10）
   - LLM 根据回归结果给出分析并生成论文摘要
5. **整合 Markdown 文档**（STEP 11）
   - 汇总所有内容生成 `paper_*.md`

执行完毕后，会在当前目录下生成一套包含图片、结果文本和 Markdown 论文的文件。

## 使用方法

确保已安装 Python3 以及 `pandas`、`matplotlib`、`statsmodels` 等依赖，在数据文件所在目录运行：

```bash
python 大模型写论文.py
```

脚本会依次使用所有模型生成论文版本。

## 许可证

本项目采用 MIT 协议发布。

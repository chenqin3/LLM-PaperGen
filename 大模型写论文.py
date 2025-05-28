# ========= workflow.py =========
import os, sys, json, subprocess, pandas as pd
from datetime import datetime

# ---------- 0. 四大模型配置 ----------
MODEL_CONFIGS = [
    {  # DeepSeek Reasoner R1
        "name": "deepseek",
        "init": lambda: __import__("openai").OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        ),
        "call": lambda cli, sys_msg, usr_msg, temp=None: cli.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":usr_msg}]
        ).choices[0].message.content
    },
    {  # o3
        "name": "o3",
        "init": lambda: __import__("openai").OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
        "call": lambda cli, sys_msg, usr_msg, temp=None: cli.chat.completions.create(
            model="o3",
            messages=[{"role":"system","content":sys_msg},
                      {"role":"user","content":usr_msg}],
            response_format={"type":"text"},
            reasoning_effort="medium"
        ).choices[0].message.content
    },
    {  # Gemini 2.5 Pro
        "name": "gemini",
        "init": lambda: __import__("google.genai", fromlist=["Client"]).Client(
            api_key=os.getenv("GEMINI_API_KEY")
        ),
        "call": lambda cli, sys_msg, usr_msg, temp=None: cli.models.generate_content(
            model="gemini-2.5-pro-preview-05-06",
            contents=f"{sys_msg}\n\n{usr_msg}"
        ).text
    },
    {  # Claude-4 Opus
        "name": "claude4",
        "init": lambda: __import__("anthropic").Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        "call": lambda cli, sys_msg, usr_msg, temp=None: cli.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            system=sys_msg,
            messages=[{"role":"user","content":[{"type":"text","text":usr_msg}]}]
        ).content[0].text
    }
]

# ---------- 1. 通用工具 ----------
def read_text(path, encodings=('utf-8','gbk','latin1')):
    for enc in encodings:
        try:
            with open(path, encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"无法读取 {path}")

def write_text(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def run_py(path):
    try:
        subprocess.check_output([sys.executable, path], stderr=subprocess.STDOUT)
        return True, ""
    except subprocess.CalledProcessError as e:
        return False, e.output.decode('utf-8', errors='ignore')

# 新增：清理LLM返回的代码字符串中的Markdown标记
def clean_llm_code_output(code_string):
    if not isinstance(code_string, str):
        return "" 

    lines = code_string.split('\n')

    # 头部清理：循环移除开头是空行或以 "```" 开头的行
    while lines:
        stripped_first_line = lines[0].strip()
        if not stripped_first_line or stripped_first_line.startswith("```"):
            lines.pop(0)
        else:
            # 找到第一个有效行，停止头部清理
            break 
    
    # 尾部清理：循环移除末尾是空行或以 "```" 开头的行
    while lines:
        stripped_last_line = lines[-1].strip()
        # 同样检查空行或以 "```" 开头的行
        if not stripped_last_line or stripped_last_line.startswith("```"):
            lines.pop(-1)
        else:
            # 找到最后一个有效行，停止尾部清理
            break
            
    return '\n'.join(lines).strip()

def fix_code(path, traceback_text, cfg, original_sys_prompt_str, original_user_msg_str, fix_request_system_prompt_str):
    bad_code = read_text(path)

    # 构建发送给 LLM (修复任务) 的 user_message
    # 这个 user_message 是一个 JSON 字符串，其结构与新的 step12_fix_code system_prompt 中描述的一致
    user_input_for_fix_task = {
        "original_task_system_prompt": original_sys_prompt_str,
        "original_task_user_input": original_user_msg_str, # original_user_msg_str 本身就是个JSON字符串
        "erroneous_code": bad_code,
        "traceback": traceback_text
    }
    # 将包含所有上下文的字典转换为JSON字符串作为用户消息
    user_msg_json_for_fix = json.dumps(user_input_for_fix_task, ensure_ascii=False, indent=2)

    cli = cfg["init"]()
    # System prompt 直接使用 prompts["step12_fix_code"]["content"] (即 fix_request_system_prompt_str)
    # User message 是上面构建的 JSON 字符串 (user_msg_json_for_fix)
    fixed_raw = cfg["call"](
        cli,
        fix_request_system_prompt_str, 
        user_msg_json_for_fix
    )
    fixed = clean_llm_code_output(fixed_raw)
    write_text(path, fixed)
    return run_py(path)

# ---------- 2. 主流程 ----------
def main():
    prompts = json.load(open('prompts.json', 'r', encoding='utf-8'))
    user_topic_input = input("▶ 请输入研究主题（留空则让模型自由发挥）：").strip()

    codebook_path, stats_path, struct_path, data_path = (
        "cfps2022codebook.csv", "summ_detail_stats.csv", "字段结构.json", "cfps2022.csv"
    )

    code_df = pd.read_csv(codebook_path, on_bad_lines='skip')
    # name_to_varlab_map: 原始字段名 -> 变量的中文标签 (来自codebook第三列varlab)
    name_to_varlab_map = dict(zip(code_df.iloc[:,0], code_df.iloc[:,2]))
    # name_to_vallab_map: 原始字段名 -> codebook中的vallab (来自codebook第二列vallab)
    name_to_vallab_map = dict(zip(code_df.iloc[:,0], code_df.iloc[:,1]))

    stats_df = pd.read_csv(stats_path, encoding='utf-8', engine='python')
    struct_json = json.load(open(struct_path, 'r', encoding='utf-8'))
    
    # 指定需要的统计特征列
    STAT_COLUMNS_TO_KEEP = ['mean', 'sd', 'min', 'p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max', 'skewness', 'kurtosis']

    for cfg in MODEL_CONFIGS:
        print(f"\n\n====================  {cfg['name'].upper()}  ====================\n")
        uid = datetime.now().strftime('%Y%m%d_%H%M%S') + "_" + cfg["name"]

        # ---------- STEP 1: Topic and Fields ----------
        # 发送给LLM的codebook抽样仅包含第一列(name)和第三列(varlab)
        cb_sample_df = pd.read_csv(codebook_path, usecols=[0, 2], on_bad_lines='skip')
        cb_sample_csv_string = cb_sample_df.to_csv(index=False)
        
        step1_user_msg = json.dumps(
            {"codebook_sample": cb_sample_csv_string, "user_topic": user_topic_input},
            ensure_ascii=False
        )
        cli = cfg["init"]()
        step1_raw = cfg["call"](
            cli,
            prompts["step1_topic_and_fields"]["content"],
            step1_user_msg
        )
        print("Step 1  输出：\\n", step1_raw, "\\n")
        cleaned_step1_raw = clean_llm_code_output(step1_raw) 


        s1 = json.loads(cleaned_step1_raw) 
        topic, need_fields = s1["topic"], s1["needed_fields"] # need_fields是原始字段名列表
        print("▶  研究主题：", topic)
        print("▶  首轮字段：", need_fields, "\n")

        # ---------- Function to build meta structure for fields ----------
        def build_fields_meta(field_names_list, codebook_df, stats_df, struct_json, name_to_varlab_map, name_to_vallab_map, stat_columns):
            meta_list = []
            # Asegurar que stats_df tenga 'variable' como índice para búsqueda eficiente
            if 'variable' not in stats_df.columns:
                print("错误：summ_detail_stats.csv 文件缺少 'variable' 列")
                return [] # O manejar el error como se prefiera
                
            stats_df_indexed = stats_df.set_index('variable', drop=False) # drop=False para mantener la columna
            
            for field_name in field_names_list:
                var_label = name_to_varlab_map.get(field_name, "") # 中文标签
                vallab_from_codebook = name_to_vallab_map.get(field_name, None) # 从codebook获取vallab
                
                value_meanings = {}
                if vallab_from_codebook:
                    value_meanings = struct_json.get(vallab_from_codebook, {}) # 用vallab去字段结构JSON查找
                else:
                    # Fallback si no se encuentra vallab, intentar con field_name (aunque la lógica solicitada es via vallab)
                    value_meanings = struct_json.get(field_name, {}) 

                field_stats_data = {}
                if field_name in stats_df_indexed.index:
                    stats_series = stats_df_indexed.loc[field_name]
                    # Filtrar solo las columnas de estadísticas deseadas y manejar NaNs
                    field_stats_data = {col: stats_series.get(col) for col in stat_columns if pd.notna(stats_series.get(col))}
                
                meta_list.append({
                    "var": field_name, # 原始字段名
                    "label": var_label, # 字段中文标签 (来自codebook的varlab)
                    "value_map": value_meanings, # 值对应的含义 (通过name -> codebook.vallab -> struct.json)
                    "stats": field_stats_data # 指定的统计特征
                })
            return meta_list

        # ---------- STEP 2 & 3: Study Plan (using new meta structure) ----------
        meta_for_study_plan = build_fields_meta(need_fields, code_df, stats_df, struct_json, name_to_varlab_map, name_to_vallab_map, STAT_COLUMNS_TO_KEEP)
        
        step3_input_data = {"topic": topic, "fields_meta": meta_for_study_plan}
        step3_user_msg = json.dumps(step3_input_data, ensure_ascii=False, sort_keys=True, indent=2) # indent for debug
        
        cli = cfg["init"]()
        step3_raw = cfg["call"](
            cli, prompts["step3_study_plan"]["content"], step3_user_msg
        )
        print("Step 3  输出：\\n", step3_raw, "\\n")
        cleaned_step3_raw = clean_llm_code_output(step3_raw) 

        s3 = json.loads(cleaned_step3_raw) 
        plan, filt_fields = s3["plan"], s3["filtered_fields"] # filt_fields是原始字段名列表
        print("▶  最终字段：", filt_fields, "\n")

        # ---------- STEP 4 & 5 & 7: Code Generation (using new meta2 structure) ----------
        meta2_for_code_gen = build_fields_meta(filt_fields, code_df, stats_df, struct_json, name_to_varlab_map, name_to_vallab_map, STAT_COLUMNS_TO_KEEP)

        # STEP 5: Generate Plot Code
        step5_input_data = {"plan": plan, "fields_meta": meta2_for_code_gen}
        step5_user_msg = json.dumps(step5_input_data, ensure_ascii=False, sort_keys=True, indent=2)
        
        cli = cfg["init"]()
        plot_code_raw = cfg["call"](cli, prompts["step5_generate_plot_code"]["content"], step5_user_msg)
        print("Step 5  生成绘图代码：\\n", plot_code_raw[:400], "...\\n")
        plot_code = clean_llm_code_output(plot_code_raw)

        plot_py = f"plot_{uid}.py"
        write_text(plot_py, plot_code)

        # Initial run for plot script
        ok, run_err = run_py(plot_py)

        if not ok:  # Initial run failed
            print(f"绘图脚本 {plot_py} 初次运行出错，错误: {str(run_err)[:200]}... 尝试第一次自动修复…")
            # First fix attempt for plot script
            ok, fix1_run_err = fix_code(
                plot_py,
                str(run_err),  # Error from initial run
                cfg,
                prompts["step5_generate_plot_code"]["content"],
                step5_user_msg,
                prompts["step12_fix_code"]["content"]
            )

            if not ok:  # First fix attempt failed
                print(f"绘图脚本 {plot_py} 第一次修复后运行仍出错，错误: {str(fix1_run_err)[:200]}... 尝试第二次自动修复…")
                # Second fix attempt for plot script
                ok, fix2_run_err = fix_code(
                    plot_py,
                    str(fix1_run_err),  # Error from run after first fix
                    cfg,
                    prompts["step5_generate_plot_code"]["content"],
                    step5_user_msg,
                    prompts["step12_fix_code"]["content"]
                )
                if not ok:  # Second fix attempt also failed
                    print(f"警告: {plot_py} 两次自动修复后仍执行失败。最后错误信息: {str(fix2_run_err)}")
        
        # Handle plot.png generation
        if os.path.exists("plot.png"):
            os.rename("plot.png", f"plot_{uid}.png")
        else:
            if ok:  # if script overall status is OK (initial or after fixes)
                print(f"注意: {plot_py} 执行成功但未生成 plot.png 文件。")
            # If not ok, the warning about failed fixes (after 2 attempts) has already been printed.

        # STEP 7: Generate Regression Code
        step7_input_data = {"plan": plan, "fields_meta": meta2_for_code_gen}
        step7_user_msg = json.dumps(step7_input_data, ensure_ascii=False, sort_keys=True, indent=2)
        
        cli = cfg["init"]()
        reg_code_raw = cfg["call"](cli, prompts["step7_generate_reg_code"]["content"], step7_user_msg)
        print("Step 7  生成回归代码：\\n", reg_code_raw[:400], "...\\n")
        reg_code = clean_llm_code_output(reg_code_raw)

        reg_py = f"reg_{uid}.py"
        write_text(reg_py, reg_code)

        # Initial run for regression script
        ok, run_err = run_py(reg_py)

        if not ok:  # Initial run failed
            print(f"回归脚本 {reg_py} 初次运行出错，错误: {str(run_err)[:200]}... 尝试第一次自动修复…")
            # First fix attempt for regression script
            ok, fix1_run_err = fix_code(
                reg_py,
                str(run_err),  # Error from initial run
                cfg,
                prompts["step7_generate_reg_code"]["content"],
                step7_user_msg,
                prompts["step12_fix_code"]["content"]
            )

            if not ok:  # First fix attempt failed
                print(f"回归脚本 {reg_py} 第一次修复后运行仍出错，错误: {str(fix1_run_err)[:200]}... 尝试第二次自动修复…")
                # Second fix attempt for regression script
                ok, fix2_run_err = fix_code(
                    reg_py,
                    str(fix1_run_err),  # Error from run after first fix
                    cfg,
                    prompts["step7_generate_reg_code"]["content"],
                    step7_user_msg,
                    prompts["step12_fix_code"]["content"]
                )
                if not ok:  # Second fix attempt also failed
                    print(f"警告: {reg_py} 两次自动修复后仍执行失败。最后错误信息: {str(fix2_run_err)}")

        # Handle result.txt generation
        if os.path.exists("result.txt"):
            os.rename("result.txt", f"result_{uid}.txt")
        else:
            if ok:  # if script overall status is OK (initial or after fixes)
                 print(f"注意: {reg_py} 执行成功但未生成 result.txt 文件。")
            # If not ok, the warning about failed fixes (after 2 attempts) has already been printed.

        # ---------- STEP 9: Result Analysis ----------
        result_txt_path = f"result_{uid}.txt"
        if not os.path.exists(result_txt_path):
            print(f"警告: 回归结果文件 {result_txt_path} 未找到，跳过结果分析和后续步骤。")
            analysis = "回归结果文件未生成，无法进行分析。"
            abstract = "由于回归结果缺失，无法生成摘要。"
        else:
            result_txt = read_text(result_txt_path)
            step9_input_data = {"plan":plan,"fields_meta":meta2_for_code_gen,"results":result_txt}
            step9_user_msg = json.dumps(step9_input_data, ensure_ascii=False, sort_keys=True, indent=2)
            cli = cfg["init"]()
            analysis = cfg["call"](cli, prompts["step9_result_analysis"]["content"], step9_user_msg)
            print("Step 9  结果分析：\n", analysis, "\n")
            write_text(f"analysis_{uid}.txt", analysis)

            # ---------- STEP 10: Abstract ----------
            cli = cfg["init"]()
            abstract = cfg["call"](cli, prompts["step10_abstract_and_intro"]["content"], analysis)
            print("Step 10  摘要：\n", abstract, "\n")
            write_text(f"abstract_{uid}.txt", abstract)

        # ---------- STEP 11: Assemble Markdown Document ----------
        md_content = []
        md_content.append(f"# {topic}\n")
        md_content.append(f"{abstract}\n")
        md_content.append("--- \n") # Page break representation or separator

        md_content.append("## 研究计划\n")
        md_content.append(f"{plan}\n")

        plot_image_path = f"plot_{uid}.png"
        if os.path.exists(plot_image_path):
            # Using relative path for the image in Markdown
            md_content.append(f"![Plot](./{os.path.basename(plot_image_path)})\n")
        else:
            md_content.append(f"[图片 {plot_image_path} 未生成]\n")

        md_content.append("\n## 回归结果\n")
        # result_txt_path was defined in STEP 7/9
        if os.path.exists(result_txt_path):
            # Wrap .txt content in a text code block to preserve formatting
            md_content.append(f"```text\n{read_text(result_txt_path)}\n```\n")
        else:
            md_content.append("[回归结果文件未生成]\n")
            
        md_content.append("\n## 结果解读\n")
        # analysis_file_path was defined in STEP 9
        if os.path.exists(f"analysis_{uid}.txt"):
            md_content.append(f"{read_text(f'analysis_{uid}.txt')}\n")
        else:
             md_content.append("[结果解读文件未生成]\n")
        
        output_md_filename = f"paper_{uid}.md"
        write_text(output_md_filename, "\n".join(md_content))
        print(f"🎉  完成：{output_md_filename}\n")

if __name__ == "__main__":
    main()

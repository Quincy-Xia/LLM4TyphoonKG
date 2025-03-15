# -*- coding: utf-8 -*-
import os

from openai import OpenAI
import pandas as pd

# Initialize DeepSeek client
client = OpenAI(api_key="Input your API", base_url="https://api.deepseek.com")

# System input (fixed part, no need to change)
system_prompt = """
You are an expert in disaster knowledge graph construction. Your task is to generate a Chain of Thought (CoT) reasoning process to explain how to transform the given input into the output. Make sure the output uses "\n" instead of actual newlines.

### **Example of Desired Chain of Thought**
Below is an example CoT for a similar task. Follow the same structure and reasoning steps when generating your response.

---
<think>\n\n好的，我现在需要解析一条关于2012年台风“杰拉华”的演化信息。目标是从原始文本中提取台风的关键发展阶段，并生成符合台风知识图谱（TyphoonKG）结构的三元组表示。\n\n### **第一步：角色设定**\n我将扮演一名 **气象知识图谱构建专家**，负责从非结构化文本中提取台风的演化状态，并按照标准化格式输出，以支持后续台风预测与分析。\n\n### **第二步：解析输入文本**\n原始输入信息如下：\n> 2012年台风杰拉华。9月20日上午7时，联合台风警报中心发布热带气旋形成警报。随后日本气象厅升为热带低压。下午2时，日本气象厅发出烈风警报。下午8时，联合台风警报中心升格为热带低压18W。\n\n我需要识别 **时间点、事件、权威机构**，并按照台风生命周期进行分类。\n\n### **第三步：时空知识补全**\n- **时间标准化**：原文本中的时间点较为简略，我需要确保格式一致，补全缺失信息。例如，“9月20日上午7时” 应补充完整为 **“2012年9月20日上午7时”**。\n- **事件划分**：识别并归类台风的不同发展阶段，如 **“形成期”** 和 **“发展期”**。\n\n### **第四步：共指消解**\n- **"联合台风警报中心"** = **JTWC (Joint Typhoon Warning Center)**，确保其统一表达方式。\n- **"日本气象厅"** = **JMA (Japan Meteorological Agency)**，避免名称歧义。\n- **"热带气旋形成警报"** 和 **"升为热带低压"** 可能表示同一事件的不同表述，需判断其语义是否相同。\n\n### **第五步：实体对齐**\n- **台风名称**：确保所有提及的“杰拉华”指向同一实体。\n- **气象状态**：对 **"热带低压18W"** 进行解析，确认其是否为编号标识。\n- **事件类型对齐**：将“升为热带低压”映射为“热带低压形成”这一标准术语。\n\n### **第六步：生命周期判定**\n台风生命周期一般分为 **形成期（Genesis）、发展期（Development）、成熟期（Mature）、衰减期（Dissipation）**，结合文本：\n1. **形成期（Genesis）**：热带气旋形成，JTWC发布警报，日本气象厅升格为热带低压。\n2. **发展期（Development）**：日本气象厅发出烈风警报，JTWC升格为热带低压18W。\n\n### **第七步：格式化输出**\n- 采用 **主语-谓语-宾语** 的三元组表示结构：\n  - **杰拉华--hasStage--形成期**\n  - **杰拉华--hasState--状态**\n  - **杰拉华--hasState--action**\n  - **杰拉华--hasState--authority**\n  - **杰拉华--hasState--time**\n- 每个演化状态对应一个独立的记录，按时间顺序排列。\n\n### **最终生成的结构化输出**\n根据上述推理过程，我得到了以下演化状态：\n\n</think>}}
---

### **Your Task**
Now, follow the same structured reasoning process for the given input and output. Provide the Chain of Thought (CoT) reasoning process in your response.
"""


# Specify folder path
folder_path = "evolution-no-cot"
output_fold_path = "evolution-with-cot"
# List all Excel files
excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]

# Create an empty DataFrame list
dfs = []

# Read each Excel file
for file in excel_files:
    file_path = os.path.join(folder_path, file)
    output_excel_file = os.path.join(output_fold_path, file)

    df = pd.read_excel(file_path, header=None)

    usage_tokens = []
    prompt_cache_hit_tokens = []
    cot_responses = []  # Store CoT responses for each record

    # Assume the Excel file has a column named "Input" containing user input text
    for index, row in df.iterrows():
        # First column as user_input
        user_input = row[0]
        # Second column as user_output
        user_output = row[1]

        # Construct messages, including only system input and current user input, without appending history
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"**Input:**\n{user_input}\n\n**Output:**\n{user_output}"}
        ]

        # Call DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            stream=False
        )

        # Get AI's reply (CoT reasoning process)
        cot_responses.append(response.choices[0].message.content.replace('\n', '\\n'))

        print(f"Processing input {index + 1}\n{'='*50}")


    # Add CoT responses to DataFrame as a new column
    df['CoT_Response'] = cot_responses

    # Output to a new Excel file
    df.to_excel(output_excel_file, index=False, header=False)  # Do not write index and column names

    print(f"Processing completed and saved to the new Excel file {output_excel_file}.")
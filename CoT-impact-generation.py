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
{<think>\n\n### **第一步：角色设定**\n好的，我现在需要解析一条关于台风“杰拉华”的演化信息，目标是标是从原始文本中提取提取灾害影响，并生成符合台风知识图谱（TyphoonKG）结构的三元组表示。\n\n### **第二步：解析输入文本**\n原始输入信息如下：\n> {台风“桑美”登陆浙江，给浙江省特别是温州、丽水两地造成了严重灾害。截至8月17日全省有18个县（市、区）的325个乡镇254.9万人受灾，3.9万间房屋倒塌，农作物受灾农田面积53.9，灾害造成的直接经济损失达到127.37亿元，因灾死亡人数193人，失踪11人。其中144人因房屋倒塌死亡。}\n\n我需要识别 **受灾对象（承灾体）、灾害事件类型（受灾实例）、相关属性，如数值信息、时间、地点**，并将其映射到标准知识图谱结构。其中受灾对象（承灾体）只能选择其中列表Categories中的一个。 Categories = ["人群"、"基础设施"、"房屋建筑"、"工业"、"公共服务设施"、"农林牧渔业"、"服务业"、"土地资源"、"生态环境资源"、"水资源"、"生物资源"、"矿产资源"] 中的一个。\n\n### **第三步：灾害事件识别**\n从输入文本中，提取涉及的 **灾害类型**，例如：\n- **受灾**（影响人口、房屋、农田等）\n- **死亡**（因灾死亡人数）\n- **失踪**（因灾失踪人数）\n- **倒塌**（房屋、建筑倒塌）\n- **损失**（经济损失）\n\n### **第四步：实体识别与标准化**\n1. **灾情对象（承灾体）** \n - **人群** → 受灾人群、死亡人群、失踪人群 \n - **房屋** → 受灾房屋、倒塌房屋 \n - **农田** → 受灾农田 \n - **财产** → 经济损失 \n\n2. **灾情事件标准化** \n - "受到台风影响，有254.9万人受灾" → **"承灾体 -- hasSubClass -- 人群 -- hasInstance -- 受灾人群 -- action -- 受灾"** \n - "因房屋倒塌死亡144人" → **"承灾体 -- hasSubClass -- 人群 -- hasInstance -- 因房屋倒塌死亡人群 -- action -- 死亡"**\n\n3. **数值提取与转换** \n - "3.9万间房屋倒塌" → **value = 39000**\n - "127.37亿元经济损失" → **value = 127.37亿元**\n\n4. **地点标准化** \n - "浙江省温州" → **location = 浙江省**\n\n### **第五步：格式化输出**\n采用 **主语-谓语-宾语** 的三元组表示结构：\n\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 受灾人群 -- action -- 受灾\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 受灾人群 -- value -- 254.9万人\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 受灾人群 -- location -- 浙江省\n\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 死亡人群 -- action -- 死亡\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 死亡人群 -- value -- 193人\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 死亡人群 -- location -- 浙江省\n\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 失踪人群 -- action -- 失踪\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 失踪人群 -- value -- 11人\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 失踪人群 -- location -- 浙江省\n\n承灾体 -- hasSubClass -- 房屋 -- hasInstance -- 受灾房屋 -- action -- 倒塌\n承灾体 -- hasSubClass -- 房屋 -- hasInstance -- 受灾房屋 -- value -- 3.9万间\n承灾体 -- hasSubClass -- 房屋 -- hasInstance -- 受灾房屋 -- location -- 浙江省\n\n承灾体 -- hasSubClass -- 农/林/牧/渔业设施/设备及产品 -- hasInstance -- 受灾农田 -- action -- 受灾\n承灾体 -- hasSubClass -- 农/林/牧/渔业设施/设备及产品 -- hasInstance -- 受灾农田 -- value -- 53.9万亩\n承灾体 -- hasSubClass -- 农/林/牧/渔业设施/设备及产品 -- hasInstance -- 受灾农田 -- location -- 浙江省\n\n承灾体 -- hasSubClass -- 财产 -- hasInstance -- 经济损失 -- action -- 损失\n承灾体 -- hasSubClass -- 财产 -- hasInstance -- 经济损失 -- value -- 127.37亿元\n承灾体 -- hasSubClass -- 财产 -- hasInstance -- 经济损失 -- location -- 浙江省\n\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 因房屋倒塌死亡人群 -- action -- 死亡\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 因房屋倒塌死亡人群 -- value -- 144人\n承灾体 -- hasSubClass -- 人群 -- hasInstance -- 因房屋倒塌死亡人群 -- location -- 浙江省\n\n### **第六步：验证输出**\n确保所有提取出的数据符合知识图谱规范，避免遗漏或格式错误。\n\n</think>
---

### **Your Task**
Now, follow the same structured reasoning process for the given input and output. Provide the Chain of Thought (CoT) reasoning process in your response.
"""


# Specify folder path
folder_path = "impact-no-cot"
output_fold_path = "impact-with-cot"
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
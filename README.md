# 中国党史RAG问答系统

针对党史领域问答场景，构建一套基于LoRA 微调大模型 + 检索增强生成（RAG）的端到端问答系统
- **github仓库**：https://github.com/u1s1-ak/RAG1 。
- **目前已实现内容**：
- **向量检索：基于本地 bge-large-zh-v1.5 模型构建 FAISS 向量库，支持 Top-K 检索 + 重排序**
- **RAG 生成：融合 LoRA 微调 Qwen3-0-6B 模型**

## 运行环境

- **平台** ： AutoDL(https://www.autodl.com/)。
- **镜像配置**： PyTorch==2.8.0 Python==3.12(ubuntu22.04) CUDA==12.8。
- **GPU** ： 5090 32G。

## 依赖安装

**执行命令**: pip install -r requirement.txt


## 数据来源

- **RAG知识库数据** : 实验采用《毛泽东邓小平江泽民胡锦涛关于中国共产党历史论述摘编 (中共中央党史和文献研究院)》以及《中国共产党简史》txt文件。
 
- **Lora微调数据获取** ： 党史 QA 数据集，数据格式为 JSON，用于LORA微调。
- **Embedding 模型** ：本地/models/bge-large-zh-v1.5，生成归一化向量。
 

## 基础模型

使用 Qwen-3-0.6B 为基础模型


## 文件作用介绍
- **1.model_download.py**:用于下载Qwen-3-0.6B模型到AutoDL本地。
- **2.generate_QA_data.py**:使用QWen3-8B模型，生成QA对，保存在data文件夹下的cpc_history_qa.json文件中。
- **3.rag-CPChistory.py**: 通过streamlit实现了一个完全本地化的中国党史领域 RAG。
- **4.Lora_train.py**: 对Qwen-3-0.6B模型进行Lora微调。
- **5.save_bge_model.py**: 用于下载/bge-large-zh-v1.5模型到AutoDL本地。
- **6.test.py**:对微调后的模型进行评估。

## 启动方式

streamlit run rag-CPChistory.py


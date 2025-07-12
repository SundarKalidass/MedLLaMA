# 🧬 MedLLaMA-LoRA-Tiny

A fine-tuned **TinyLLaMA model** on the **PubMedQA** dataset using **LoRA** for healthcare-specific Question Answering.

---

## 🚀 Project Overview

MedLLaMA-LoRA-Tiny is a lightweight healthcare domain language model designed for:

* ✅ Medical Question Answering
* ✅ Fine-tuned performance on PubMedQA
* ✅ Efficient deployment using LoRA
* ✅ Suitable for RAG (Retrieval-Augmented Generation)

---

## 🧪 Dataset

**🔬 PubMedQA** is a biomedical QA dataset built using PubMed abstracts.

* Each sample has a question, an abstract, and a yes/no/maybe answer.
* Ideal for training LLMs for real-world clinical and biomedical understanding.

**Dataset Link:** [PubMedQA (HuggingFace)](https://huggingface.co/datasets/pubmed_qa)

---

## 🏗️ Model Architecture

* **Base Model:** [`TinyLLaMA/TinyLlama-1.1B-Chat`](https://huggingface.co/cspray/TinyLlama-1.1B-Chat)
* **Fine-Tuning Strategy:** LoRA (Low-Rank Adaptation)
* **Training Platform:** Google Colab

---

## 🧠 How It Works

The model has been trained to:

* Understand medical queries
* Extract clinical context
* Generate accurate and medically-aligned answers

You can combine this model later with vector-based document retrieval (like FAISS) to enable RAG-based intelligent search.

---

## 📁 Files Included

* `medllama_lora_model/` — Fine-tuned model weights (adapter\_model.bin, config.json, tokenizer, etc.)
* `medllama_finetune.ipynb` — Google Colab training notebook
* `README.md` — This file

---

## 💬 Example Prompts

#### 🩺 Example 1

**Prompt:**

> Does aspirin reduce heart attack risk?

**Model Answer:**

> Yes, aspirin can reduce heart attack risk by preventing blood clots, but should be used under medical advice.

---

#### 👶 Example 2

**Prompt:**

> Is paracetamol safe during pregnancy?

**Model Answer:**

> Paracetamol is generally safe during pregnancy when used in recommended doses.

---

## 🧠 Future Work

* ✅ Integrate with FastAPI for real-time inference
* ✅ Enable vector retrieval from real patient documents
* ✅ Deploy on Hugging Face Spaces / Streamlit

---

## 🤝 Contribution

Pull requests and feedback are welcome. If you use this model or build on it, kindly give credit.

---

## 📜 License

This project is licensed under the MIT License.

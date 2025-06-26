# EVENTA Grand Challenge – Track 1: Event-Enriched Image Captioning

This repository contains our submission to the **EVENTA Grand Challenge - ACM Multimedia 2025**, **Track 1: Event-Enriched Image Captioning**.

Our system aims to generate context-aware image captions by leveraging both the **visual content** and the **accompanying news article**, with a focus on maximizing **semantic richness** and **CIDEr score**.

---

## 🧠 Overview

We adopt a **two-stage captioning pipeline** that integrates image and article information in a controlled manner. The system is designed to:

- Utilize high-context transformer models with **extended input lengths** (up to 8K tokens)
- Combine **visual content**, **article context**, and **semantic features**
- Apply **prompt engineering** and structured guidance to improve output quality

---

## 📌 Approach

### Stage 1: Caption Generation

- **Model**: LLaVA 1.6 Mistral 7B (via Hugging Face, long context support)
- **Input**: Raw image and full article content
- **Enhancement**: Semantic signals derived from the article (e.g. keywords, tags)
- **Goal**: Produce an initial caption that reflects both the image and event context

### Stage 2: Caption Refinement (Optional)

- **Model**: Large language model with 8K+ context window (e.g. Mixtral 8x7B Instruct)
- **Input**: [Image] + [Article] + [Initial Caption]
- **Purpose**: Improve linguistic fluency, factual alignment, and semantic depth

---

## 🔍 Semantic Extraction

We extract auxiliary **semantic tags** from the article using lightweight techniques, such as:

- TF-IDF keyword scoring
- Named Entity Recognition (NER)
- Optionally: domain-specific tag dictionaries

These signals are injected into the prompt construction process (but not disclosed here), serving as high-level guidance for generation.

---

## 🧪 Evaluation

We evaluate the model's output primarily using the **CIDEr metric**, while monitoring BLEU, METEOR, and SPICE for completeness.

---

## 🛠️ Requirements

```bash
pip install torch transformers pillow protobuf scikit-learn
  

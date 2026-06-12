# Benchmark · Modelos de Embeddings para RAG de RRHH

**Fecha:** 2026-06-10 23:45  
**Modelos evaluados:** all-MiniLM-L6-v2 · multilingual-e5-small · minilm-code-search-512  
**Casos evaluados:** 16  
**Corpus:** `data\procesado\HalemoGPA_processed.jsonl` + 2 documentos sintéticos (LangChain, LangGraph)  
**Formato documental:** `Skill: … | Evidence: … | Code: …`  

---

## Resumen global

| Modelo | Hit@1 | MRR | P@3 | R@3 | Avg Gap | ms/enc | Score |
|--------|-------|-----|-----|-----|---------|--------|-------|
| 🥇 `minilm-code-search-512` | 93.8% | 0.953 | 0.562 | 0.791 | +0.1553 | 11.1ms | **0.8356** |
| `all-MiniLM-L6-v2` | 87.5% | 0.896 | 0.542 | 0.728 | +0.1949 | 11.6ms | **0.8088** |
| `multilingual-e5-small` | 87.5% | 0.927 | 0.604 | 0.869 | +0.0380 | 22.5ms | **0.7647** |

> **Score** = 0.35 × Hit@1 + 0.25 × MRR + 0.20 × P@3 + 0.10 × R@3 + 0.10 × norm(AvgGap)  
> norm(AvgGap) = min(max(AvgGap × 5, 0), 1)

**Modelo ganador:** `minilm-code-search-512` con score 0.8356

---

## Análisis por tipo de caso

| Tipo | Modelo | Hit@1 | MRR | P@3 | Avg Gap | N casos |
|------|--------|-------|-----|-----|---------|---------|
| code_only | `all-MiniLM-L6-v2` | 100% | 1.000 | 1.000 | +0.1372 | 1 |
| code_only | `multilingual-e5-small` | 100% | 1.000 | 1.000 | +0.0466 | 1 |
| code_only | `minilm-code-search-512` | 100% | 1.000 | 1.000 | +0.0592 | 1 |
| cross_lingual | `all-MiniLM-L6-v2` | 100% | 1.000 | 0.667 | +0.4125 | 1 |
| cross_lingual | `multilingual-e5-small` | 100% | 1.000 | 0.667 | +0.0516 | 1 |
| cross_lingual | `minilm-code-search-512` | 100% | 1.000 | 0.667 | +0.3292 | 1 |
| false_positive_trap | `all-MiniLM-L6-v2` | 100% | 1.000 | 1.000 | +0.4751 | 1 |
| false_positive_trap | `multilingual-e5-small` | 100% | 1.000 | 1.000 | +0.0340 | 1 |
| false_positive_trap | `minilm-code-search-512` | 100% | 1.000 | 1.000 | +0.3271 | 1 |
| profile_query | `all-MiniLM-L6-v2` | 100% | 1.000 | 0.667 | +0.2302 | 1 |
| profile_query | `multilingual-e5-small` | 100% | 1.000 | 0.667 | +0.0724 | 1 |
| profile_query | `minilm-code-search-512` | 100% | 1.000 | 0.667 | +0.2004 | 1 |
| rag_real_format | `all-MiniLM-L6-v2` | 80% | 0.833 | 0.467 | +0.1735 | 10 |
| rag_real_format | `multilingual-e5-small` | 90% | 0.933 | 0.567 | +0.0403 | 10 |
| rag_real_format | `minilm-code-search-512` | 100% | 1.000 | 0.533 | +0.1528 | 10 |
| synthetic_agents | `all-MiniLM-L6-v2` | 100% | 1.000 | 0.333 | +0.0115 | 1 |
| synthetic_agents | `multilingual-e5-small` | 0% | 0.500 | 0.333 | -0.0190 | 1 |
| synthetic_agents | `minilm-code-search-512` | 100% | 1.000 | 0.333 | +0.0829 | 1 |
| synthetic_rag | `all-MiniLM-L6-v2` | 100% | 1.000 | 0.333 | +0.1164 | 1 |
| synthetic_rag | `multilingual-e5-small` | 100% | 1.000 | 0.333 | +0.0198 | 1 |
| synthetic_rag | `minilm-code-search-512` | 0% | 0.250 | 0.000 | -0.0428 | 1 |

---

## Detalle por caso

### C01 · IA médica con PyTorch

**Tipo:** `rag_real_format`  
**Query:** _Busco un perfil con experiencia en modelos de IA para clasificación de imágenes médicas MRI_  
**Nota:** Debe recuperar evidencias de PyTorch relacionadas con clasificación de tumores MRI.  
**Skills relevantes:** `PyTorch`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 1.000 | 0.200 | 0.6252 | 0.3033 | +0.3218 | 12.4ms |
| `multilingual-e5-small` | ✓ | 1.000 | 1.000 | 0.200 | 0.8829 | 0.8333 | +0.0496 | 21.7ms |
| `minilm-code-search-512` | ✓ | 1.000 | 1.000 | 0.200 | 0.5674 | 0.4894 | +0.0780 | 12.4ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PyTorch` | 0.6252 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `PyTorch` | 0.5931 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `PyTorch` | 0.5920 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PyTorch` | 0.8829 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `PyTorch` | 0.8762 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `PyTorch` | 0.8750 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PyTorch` | 0.5674 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `PyTorch` | 0.5542 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `PyTorch` | 0.5476 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C02 · Interfaz web para aplicación de IA

**Tipo:** `rag_real_format`  
**Query:** _Necesito alguien que pueda crear una interfaz web interactiva para mostrar resultados de un modelo de IA_  
**Nota:** Debe recuperar Streamlit frente a librerías auxiliares.  
**Skills relevantes:** `Streamlit`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.667 | 0.500 | 0.5390 | 0.3940 | +0.1450 | 11.3ms |
| `multilingual-e5-small` | ✓ | 1.000 | 1.000 | 0.750 | 0.8788 | 0.8326 | +0.0462 | 18.9ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.667 | 0.500 | 0.4383 | 0.3634 | +0.0749 | 10.1ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Streamlit` | 0.5390 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Streamlit` | 0.4061 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `urllib3` | 0.3940 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Streamlit` | 0.8788 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Streamlit` | 0.8543 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `Streamlit` | 0.8485 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Streamlit` | 0.4383 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Streamlit` | 0.3649 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `JavaScript` | 0.3634 | HalemoGPA/Learn-Js | file |

### C03 · Explicabilidad visual en visión por computador

**Tipo:** `rag_real_format`  
**Query:** _Busco experiencia en explicabilidad visual para redes neuronales de imágenes usando mapas de calor_  
**Nota:** Debe recuperar Grad-CAM.  
**Skills relevantes:** `Grad-CAM`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.667 | 1.000 | 0.5284 | 0.3737 | +0.1547 | 10.3ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.667 | 1.000 | 0.8971 | 0.8386 | +0.0585 | 22.3ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.667 | 1.000 | 0.5275 | 0.4538 | +0.0737 | 9.6ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Grad-CAM` | 0.5284 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Grad-CAM` | 0.4112 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.3737 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Grad-CAM` | 0.8971 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Grad-CAM` | 0.8781 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.8386 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Grad-CAM` | 0.5275 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Grad-CAM` | 0.5046 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.4538 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C04 · Procesamiento de imágenes con OpenCV

**Tipo:** `rag_real_format`  
**Query:** _Necesito experiencia procesando imágenes con OpenCV en Python_  
**Nota:** Debe distinguir OpenCV de PyTorch.  
**Skills relevantes:** `OpenCV`, `OpenCV-Python-Headless`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.667 | 1.000 | 0.7495 | 0.6018 | +0.1477 | 9.6ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.667 | 1.000 | 0.9031 | 0.8884 | +0.0146 | 20.7ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.667 | 1.000 | 0.7333 | 0.6319 | +0.1015 | 8.9ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `OpenCV-Python-Headless` | 0.7495 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `OpenCV` | 0.6706 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.6018 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `OpenCV-Python-Headless` | 0.9031 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `OpenCV` | 0.8969 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.8884 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `OpenCV-Python-Headless` | 0.7333 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `OpenCV` | 0.7192 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.6319 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C05 · Despliegue con Docker

**Tipo:** `false_positive_trap`  
**Query:** _Busco un desarrollador con experiencia creando contenedores Docker y Dockerfile para desplegar aplicaciones_  
**Nota:** Debe priorizar Docker frente a despliegue genérico o Streamlit Cloud.  
**Skills relevantes:** `Docker`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 1.000 | 0.750 | 0.6756 | 0.2004 | +0.4751 | 8.1ms |
| `multilingual-e5-small` | ✓ | 1.000 | 1.000 | 0.750 | 0.8878 | 0.8538 | +0.0340 | 17.9ms |
| `minilm-code-search-512` | ✓ | 1.000 | 1.000 | 0.750 | 0.5911 | 0.2640 | +0.3271 | 7.5ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Docker` | 0.6756 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Docker` | 0.6253 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `Docker` | 0.5951 | HalemoGPA/HalemoGPA | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Docker` | 0.8878 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Docker` | 0.8853 | HalemoGPA/HalemoGPA | commit |
| #3 | ✓ REL | `Docker` | 0.8783 | HalemoGPA/HalemoGPA | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Docker` | 0.5911 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `Docker` | 0.5296 | HalemoGPA/HalemoGPA | commit |
| #3 | ✓ REL | `Docker` | 0.4965 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C06 · Backend API con FastAPI

**Tipo:** `rag_real_format`  
**Query:** _Necesito desarrollar una API backend en Python con FastAPI_  
**Nota:** Debe recuperar FastAPI.  
**Skills relevantes:** `FastAPI`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.333 | 1.000 | 0.7046 | 0.4443 | +0.2603 | 12.2ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.333 | 1.000 | 0.9119 | 0.8429 | +0.0690 | 23.5ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.333 | 1.000 | 0.5579 | 0.3208 | +0.2371 | 11.1ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `FastAPI` | 0.7046 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Docker` | 0.4443 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `urllib3` | 0.4437 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `FastAPI` | 0.9119 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Docker` | 0.8429 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `PyTorch` | 0.8414 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `FastAPI` | 0.5579 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Docker` | 0.3208 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `urllib3` | 0.2363 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C07 · Frontend con React y TypeScript

**Tipo:** `profile_query`  
**Query:** _Busco un perfil frontend con experiencia en React, componentes y tipado con TypeScript_  
**Nota:** Debe recuperar evidencias frontend.  
**Skills relevantes:** `React`, `TypeScript`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.667 | 1.000 | 0.4796 | 0.2493 | +0.2302 | 12.5ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.667 | 1.000 | 0.9044 | 0.8321 | +0.0724 | 24.5ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.667 | 1.000 | 0.4076 | 0.2071 | +0.2004 | 12.3ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `TypeScript` | 0.4796 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `React` | 0.4476 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Docker` | 0.2493 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `TypeScript` | 0.9044 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `React` | 0.8912 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Docker` | 0.8321 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `React` | 0.4076 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `TypeScript` | 0.4033 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Docker` | 0.2071 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C08 · Testing de componentes React

**Tipo:** `rag_real_format`  
**Query:** _Necesito experiencia escribiendo tests unitarios de componentes React con Jest y React Testing Library_  
**Nota:** Debe recuperar testing real, no solo frontend genérico.  
**Skills relevantes:** `Jest`, `React Testing Library`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.667 | 1.000 | 0.6539 | 0.3351 | +0.3188 | 14.4ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.667 | 1.000 | 0.9250 | 0.8535 | +0.0715 | 26.6ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.667 | 1.000 | 0.6481 | 0.3406 | +0.3075 | 13.9ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `React Testing Library` | 0.6539 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `Jest` | 0.6138 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Redux` | 0.3351 | HalemoGPA/HalemoGPA | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Jest` | 0.9250 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `React Testing Library` | 0.9024 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `JavaScript` | 0.8535 | HalemoGPA/HalemoGPA | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `React Testing Library` | 0.6481 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `Jest` | 0.5728 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Redux` | 0.3406 | HalemoGPA/HalemoGPA | commit |

### C09 · Base de datos SQL en Python

**Tipo:** `rag_real_format`  
**Query:** _Busco experiencia conectando aplicaciones Python con bases de datos SQL PostgreSQL_  
**Nota:** Debe recuperar PostgreSQL y psycopg2.  
**Skills relevantes:** `PostgreSQL`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.333 | 1.000 | 0.6601 | 0.2478 | +0.4123 | 11.1ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.333 | 1.000 | 0.8989 | 0.8515 | +0.0474 | 23.7ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.333 | 1.000 | 0.5810 | 0.2000 | +0.3810 | 10.9ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PostgreSQL` | 0.6601 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Streamlit` | 0.2478 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Streamlit` | 0.2475 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PostgreSQL` | 0.8989 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Streamlit` | 0.8515 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Streamlit` | 0.8482 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PostgreSQL` | 0.5810 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Streamlit` | 0.2000 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `OpenCV` | 0.1883 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C10 · Generación de informes PDF

**Tipo:** `rag_real_format`  
**Query:** _Necesito generar informes PDF personalizados desde Python_  
**Nota:** Debe recuperar reportlab.  
**Skills relevantes:** `reportlab`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.333 | 1.000 | 0.6186 | 0.3532 | +0.2654 | 10.3ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.333 | 1.000 | 0.8782 | 0.8527 | +0.0255 | 20.2ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.333 | 1.000 | 0.6070 | 0.3899 | +0.2171 | 10.3ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `reportlab` | 0.6186 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✗ IRR | `Docker` | 0.3532 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Streamlit` | 0.3465 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `reportlab` | 0.8782 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✗ IRR | `PyTorch` | 0.8527 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Docker` | 0.8441 | HalemoGPA/HalemoGPA | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `reportlab` | 0.6070 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✗ IRR | `PyTorch` | 0.3899 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Docker` | 0.3696 | HalemoGPA/HalemoGPA | commit |

### C11 · Peticiones HTTP desde frontend

**Tipo:** `rag_real_format`  
**Query:** _Busco experiencia haciendo peticiones HTTP desde una aplicación frontend_  
**Nota:** Debe distinguir Axios (frontend) de requests/urllib3 (backend).  
**Skills relevantes:** `Axios`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✗ | 0.167 | 0.000 | 0.000 | 0.2461 | 0.4403 | -0.1942 | 12.4ms |
| `multilingual-e5-small` | ✗ | 0.333 | 0.333 | 1.000 | 0.8522 | 0.8557 | -0.0035 | 23.4ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.333 | 1.000 | 0.3228 | 0.2750 | +0.0479 | 11.4ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✗ IRR | `requests` | 0.4403 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✗ IRR | `urllib3` | 0.3438 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `HF Spaces` | 0.3049 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✗ IRR | `urllib3` | 0.8557 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✗ IRR | `requests` | 0.8548 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `Axios` | 0.8522 | HalemoGPA/HalemoGPA | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Axios` | 0.3228 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `requests` | 0.2750 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `urllib3` | 0.2587 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C12 · Gestión de estado frontend

**Tipo:** `rag_real_format`  
**Query:** _Necesito experiencia gestionando estado global en aplicaciones frontend_  
**Nota:** Debe recuperar Redux frente a otras librerías frontend.  
**Skills relevantes:** `Redux`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✗ | 0.167 | 0.000 | 0.000 | 0.1777 | 0.2749 | -0.0972 | 12.3ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.333 | 1.000 | 0.8639 | 0.8401 | +0.0238 | 22.9ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.333 | 1.000 | 0.3464 | 0.3372 | +0.0092 | 11.8ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✗ IRR | `React` | 0.2749 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `CSS` | 0.2317 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `FastAPI` | 0.2308 | HalemoGPA/HalemoGPA | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Redux` | 0.8639 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `FastAPI` | 0.8401 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `uvicorn` | 0.8319 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `Redux` | 0.3464 | HalemoGPA/HalemoGPA | commit |
| #2 | ✗ IRR | `Axios` | 0.3372 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `React Router` | 0.3134 | HalemoGPA/HalemoGPA | commit |

### C13 · Sistema RAG con base vectorial y LLM

**Tipo:** `synthetic_rag`  
**Query:** _Busco un desarrollador para crear sistemas RAG con embeddings, bases vectoriales y LLMs_  
**Nota:** Caso central del TFG. Documento sintético.  
**Skills relevantes:** `LangChain`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.333 | 1.000 | 0.4078 | 0.2914 | +0.1164 | 10.3ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.333 | 1.000 | 0.8661 | 0.8463 | +0.0198 | 21.0ms |
| `minilm-code-search-512` | ✗ | 0.250 | 0.000 | 0.000 | 0.2697 | 0.3125 | -0.0428 | 10.4ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `LangChain` | 0.4078 | synthetic/rag-demo | file |
| #2 | ✗ IRR | `Streamlit` | 0.2914 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Streamlit` | 0.2044 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `LangChain` | 0.8661 | synthetic/rag-demo | file |
| #2 | ✗ IRR | `PyTorch` | 0.8463 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Streamlit` | 0.8339 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✗ IRR | `Streamlit` | 0.3125 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✗ IRR | `PyTorch` | 0.3012 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `PyTorch` | 0.2737 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C14 · Agentes de IA y workflows multiagente

**Tipo:** `synthetic_agents`  
**Query:** _Busco un perfil para desarrollar agentes de IA con herramientas y flujos multiagente_  
**Nota:** Comprueba si el modelo relaciona agentes IA con LangGraph. Documento sintético.  
**Skills relevantes:** `LangGraph`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.333 | 1.000 | 0.2723 | 0.2608 | +0.0115 | 10.2ms |
| `multilingual-e5-small` | ✗ | 0.500 | 0.333 | 1.000 | 0.8250 | 0.8440 | -0.0190 | 19.4ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.333 | 1.000 | 0.3039 | 0.2209 | +0.0829 | 9.8ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `LangGraph` | 0.2723 | synthetic/agent-demo | file |
| #2 | ✗ IRR | `Streamlit` | 0.2608 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Streamlit` | 0.2362 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✗ IRR | `Streamlit` | 0.8440 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `LangGraph` | 0.8250 | synthetic/agent-demo | file |
| #3 | ✗ IRR | `Docker` | 0.8247 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `LangGraph` | 0.3039 | synthetic/agent-demo | file |
| #2 | ✗ IRR | `Docker` | 0.2209 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✗ IRR | `Streamlit` | 0.2199 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C15 · Código puro: PyTorch training

**Tipo:** `code_only`  
**Query:** _deep learning model training with PyTorch_  
**Nota:** Observar si el modelo de code search mejora cuando la query es en inglés y técnica.  
**Skills relevantes:** `PyTorch`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 1.000 | 0.200 | 0.4686 | 0.3314 | +0.1372 | 13.9ms |
| `multilingual-e5-small` | ✓ | 1.000 | 1.000 | 0.200 | 0.8541 | 0.8075 | +0.0466 | 26.5ms |
| `minilm-code-search-512` | ✓ | 1.000 | 1.000 | 0.200 | 0.5569 | 0.4977 | +0.0592 | 13.5ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PyTorch` | 0.4686 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `PyTorch` | 0.4353 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `PyTorch` | 0.4304 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PyTorch` | 0.8541 | HalemoGPA/HalemoGPA | commit |
| #2 | ✓ REL | `PyTorch` | 0.8473 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `PyTorch` | 0.8459 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `PyTorch` | 0.5569 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #2 | ✓ REL | `PyTorch` | 0.5332 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |
| #3 | ✓ REL | `PyTorch` | 0.5101 | HalemoGPA/BrainMRI-Tumor-Classifier-Pytorch | commit |

### C16 · Cross-lingual: JavaScript y DOM

**Tipo:** `cross_lingual`  
**Query:** _Busco experiencia desarrollando aplicaciones web interactivas con JavaScript y manipulación del DOM_  
**Nota:** Consulta en español frente a evidencias en inglés/español. Prueba ventaja de multilingual-e5.  
**Skills relevantes:** `JavaScript`  

| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |
|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|
| `all-MiniLM-L6-v2` | ✓ | 1.000 | 0.667 | 1.000 | 0.4883 | 0.0758 | +0.4125 | 14.4ms |
| `multilingual-e5-small` | ✓ | 1.000 | 0.667 | 1.000 | 0.8658 | 0.8142 | +0.0516 | 26.5ms |
| `minilm-code-search-512` | ✓ | 1.000 | 0.667 | 1.000 | 0.5363 | 0.2071 | +0.3292 | 14.1ms |

**Top-3 recuperado:**

*all-MiniLM-L6-v2*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `JavaScript` | 0.4883 | HalemoGPA/Learn-Js | file |
| #2 | ✓ REL | `JavaScript` | 0.4180 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Docker` | 0.0758 | HalemoGPA/HalemoGPA | commit |

*multilingual-e5-small*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `JavaScript` | 0.8658 | HalemoGPA/Learn-Js | file |
| #2 | ✓ REL | `JavaScript` | 0.8471 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `Docker` | 0.8142 | HalemoGPA/HalemoGPA | commit |

*minilm-code-search-512*

| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |
|------|-----------|-------|-----|------|----------------|
| #1 | ✓ REL | `JavaScript` | 0.5363 | HalemoGPA/Learn-Js | file |
| #2 | ✓ REL | `JavaScript` | 0.4403 | HalemoGPA/HalemoGPA | commit |
| #3 | ✗ IRR | `PostgreSQL` | 0.2071 | HalemoGPA/HalemoGPA | commit |

---

## Conclusión automática

El modelo con mejor score es **`minilm-code-search-512`** (score=0.8356, Δ=+0.0709 respecto al siguiente). La ventaja es significativa respecto al resto de modelos.

El rendimiento de `minilm-code-search-512` puede deberse a que está afinado para búsqueda semántica de código, lo que le permite aprovechar mejor el campo `Code` presente en los documentos. Aunque es un modelo ligero, su especialización puede compensar la falta de soporte multilingüe en consultas en español.

**Advertencia de alcance:** estos resultados dependen del corpus reducido (16 casos de prueba, corpus de menos de 50 evidencias reales más 2 documentos sintéticos), las consultas definidas manualmente y el formato documental del prototipo. No deben interpretarse como una evaluación universal de los modelos. Su propósito es justificar la elección del modelo de embeddings para ChromaDB en el contexto concreto del sistema RAG del TFG.

---
*Generado automáticamente por `eval_embeddings.py`*
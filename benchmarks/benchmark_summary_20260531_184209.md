# Benchmark · Extracción de skills técnicas

**Fecha:** 2026-05-31 18:08  
**Modelos evaluados:** `qwen2.5-coder:7b`, `qwen3:4b`, `phi4-mini:3.8b`, `codegemma:7b-instruct`, `granite-code:8b`  
**Casos de prueba:** 8 (7 evaluados + 1 ambiguo)  

---

## Resumen global

| Modelo | Parse OK | Precision | Recall | F1 | Alucinaciones | Media/caso | Tiempo total | Score |
|--------|----------|-----------|--------|----|---------------|------------|--------------|-------|
| 🥇`qwen2.5-coder:7b` | 7/7 | 54.8% | 66.7% | 64.4% | 🟢 0 | 28.7s | 3.6 min | **0.67** |
| `qwen3:4b` | 7/7 | 69.4% | 80.6% | 71.1% | 🔴 4 | 110.9s | 13.8 min | **0.66** |
| `granite-code:8b` | 7/7 | 46.4% | 58.3% | 55.9% | 🟢 0 | 35.9s | 4.6 min | **0.59** |
| `codegemma:7b-instruct` | 7/7 | 28.1% | 66.7% | 43.0% | 🟢 0 | 64.9s | 8.5 min | **0.50** |
| `phi4-mini:3.8b` | 6/7 | 19.0% | 33.3% | 26.2% | 🟢 0 | 28.9s | 3.6 min | **0.34** |

> Score = 0.50×F1 + 0.20×Precision + 0.20×Recall + 0.10×(sin alucinaciones)

---

## Detalle por caso

| Caso | Tipo | Modelo | Detectadas | P | R | F1 | Alucin. | s |
|------|------|--------|------------|---|---|----|---------|---|
| C1 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 50.0% | 33.3% | 40.0% | 🟢 | 29.9 |
| C1 | true_positive | `qwen3:4b` | ✓ LangChain, LangGraph | 100.0% | 66.7% | 80.0% | 🟢 | 137.2 |
| C1 | true_positive | `phi4-mini:3.8b` | ✓ FastAPI, LangChainOpenAI, LangGraph, ToolNode | 50.0% | 66.7% | 57.1% | 🟢 | 51.7 |
| C1 | true_positive | `codegemma:7b-instruct` | ✓ ChatOpenAI, Docker, FastAPI, LangChain, PyTorch | 20.0% | 33.3% | 25.0% | 🟢 | 76.6 |
| C1 | true_positive | `granite-code:8b` | ✓ Docker, FastAPI, LangChain, PyTorch | 25.0% | 33.3% | 28.6% | 🟢 | 60.4 |
| C2 | comment_trap | `qwen2.5-coder:7b` | ✓ NumPy, Pandas | 100.0% | 100.0% | 100.0% | 🟢 | 44.2 |
| C2 | comment_trap | `qwen3:4b` | ✓ LangChain, LangGraph, NumPy, OpenAI API, Pandas, RAG | 33.3% | 100.0% | 50.0% | 🔴 4 | 170.8 |
| C2 | comment_trap | `phi4-mini:3.8b` | ✓ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 30.7 |
| C2 | comment_trap | `codegemma:7b-instruct` | ✓ NumPy, NumPy matrix, Pandas, Resampling, Z-score normalization | 40.0% | 100.0% | 57.1% | 🟢 | 72.9 |
| C2 | comment_trap | `granite-code:8b` | ✓ numpy, pandas | 100.0% | 100.0% | 100.0% | 🟢 | 41.0 |
| C3 | general | `qwen2.5-coder:7b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 41.4 |
| C3 | general | `qwen3:4b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 133.0 |
| C3 | general | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 53.4 |
| C3 | general | `codegemma:7b-instruct` | ✓ Docker, FastAPI, Pydantic, SQLAlchemy | 75.0% | 100.0% | 85.7% | 🟢 | 62.5 |
| C3 | general | `granite-code:8b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 27.9 |
| C4 | ambiguo | `qwen2.5-coder:7b` | LangChain | — | — | — | — | 13.8 |
| C4 | ambiguo | `qwen3:4b` | LangChain | — | — | — | — | 52.9 |
| C4 | ambiguo | `phi4-mini:3.8b` | ChatOpenAI, LangChainOpenAI | — | — | — | — | 15.6 |
| C4 | ambiguo | `codegemma:7b-instruct` | ChatOpenAI, GPT-4o, HumanMessage, LangChain, Python | — | — | — | — | 54.7 |
| C4 | ambiguo | `granite-code:8b` | ChatOpenAI | — | — | — | — | 24.2 |
| C5 | true_positive | `qwen2.5-coder:7b` | ✓ PyTorch, Transformers, datasets | 66.7% | 100.0% | 80.0% | 🟢 | 23.7 |
| C5 | true_positive | `qwen3:4b` | ✓ Datasets, PyTorch, Transformers | 66.7% | 100.0% | 80.0% | 🟢 | 91.3 |
| C5 | true_positive | `phi4-mini:3.8b` | ✓ Datasets, Hugging Face's Transformers, PyTorch Lightning/Datasets/Trainer, Transformers | 50.0% | 100.0% | 66.7% | 🟢 | 16.8 |
| C5 | true_positive | `codegemma:7b-instruct` | ✓ CUDA, Datasets, PyTorch, Trainer, TrainingArguments, Transformers | 33.3% | 100.0% | 50.0% | 🟢 | 60.5 |
| C5 | true_positive | `granite-code:8b` | ✓ Docker, Hugging Face Transformers, PyTorch | 33.3% | 50.0% | 40.0% | 🟢 | 32.9 |
| C6 | comment_trap | `qwen2.5-coder:7b` | ✓ CSS Clamp Function, Responsive Design | 0.0% | n/a | n/a | 🟢 | 19.1 |
| C6 | comment_trap | `qwen3:4b` | ✓ ∅ | n/a | n/a | n/a | 🟢 | 37.8 |
| C6 | comment_trap | `phi4-mini:3.8b` | ✓ CSS clamp(), z-index | 0.0% | n/a | n/a | 🟢 | 15.0 |
| C6 | comment_trap | `codegemma:7b-instruct` | ✓ CSS Grid, Clamp CSS, Z-index | 0.0% | n/a | n/a | 🟢 | 47.3 |
| C6 | comment_trap | `granite-code:8b` | ✓ clamp(), grid-template-columns, z-index | 0.0% | n/a | n/a | 🟢 | 32.4 |
| C7 | true_positive | `qwen2.5-coder:7b` | ✓ Anthropic, ChromaDB, SentenceTransformer | 66.7% | 66.7% | 66.7% | 🟢 | 23.5 |
| C7 | true_positive | `qwen3:4b` | ✓ Anthropic, ChromaDB, SentenceTransformers | 66.7% | 66.7% | 66.7% | 🟢 | 155.8 |
| C7 | true_positive | `phi4-mini:3.8b` | ✓ AnthropicAI, ChromaDB, SentenceTransformer | 33.3% | 33.3% | 33.3% | 🟢 | 16.3 |
| C7 | true_positive | `codegemma:7b-instruct` | ✓ Anthropic, ChromaDB, Docker, FastAPI, LangChain, PyTorch, SentenceTransformer | 28.6% | 66.7% | 40.0% | 🟢 | 71.4 |
| C7 | true_positive | `granite-code:8b` | ✓ SentenceTransformer, anthropic, chromadb | 66.7% | 66.7% | 66.7% | 🟢 | 28.6 |
| C8 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 0.0% | 0.0% | 0.0% | 🟢 | 18.9 |
| C8 | true_positive | `qwen3:4b` | ✓ CrewAI, LangChain | 50.0% | 50.0% | 50.0% | 🟢 | 50.1 |
| C8 | true_positive | `phi4-mini:3.8b` | ✓ Docker, FastAPI, LangChain, PyTorch | 0.0% | 0.0% | 0.0% | 🟢 | 18.8 |
| C8 | true_positive | `codegemma:7b-instruct` | ✓ ChatOpenAI, Docker, FastAPI, LangChain, PyTorch | 0.0% | 0.0% | 0.0% | 🟢 | 63.4 |
| C8 | true_positive | `granite-code:8b` | ✓ ChatOpenAI, FastAPI, LangChain | 0.0% | 0.0% | 0.0% | 🟢 | 28.4 |

---

## Conclusión

### Modelo recomendado: `qwen2.5-coder:7b`

Con un score de **0.67**, `qwen2.5-coder:7b` obtiene el mejor balance entre precisión (54.8%), recall (66.7%) y F1 (64.4%).

La ventaja sobre el segundo clasificado es ajustada (Δ={score_gap:.2f}) — ambos son opciones viables.

**Alucinaciones:** Solo `qwen2.5-coder:7b`, `granite-code:8b`, `codegemma:7b-instruct`, `phi4-mini:3.8b` evitó las alucinaciones.

**Latencia:** el modelo más rápido es `qwen2.5-coder:7b` con 28.7s de media por llamada (3.6 min en total para los 8 casos).

### Caso ambiguo (C4)

El caso C4 (import sin uso real) revela la agresividad de cada modelo:

- `qwen2.5-coder:7b`: LangChain — ⚠️ agresivo
- `qwen3:4b`: LangChain — ⚠️ agresivo
- `phi4-mini:3.8b`: ChatOpenAI, LangChainOpenAI — ⚠️ agresivo
- `codegemma:7b-instruct`: ChatOpenAI, GPT-4o, HumanMessage, LangChain, Python — ⚠️ agresivo
- `granite-code:8b`: ChatOpenAI — ⚠️ agresivo

Un modelo conservador (∅ o pocos resultados en C4) es preferible para este sistema: reduce el ruido en el perfil técnico del usuario.

---
*Generado automáticamente por `benchmark_models.py`*
# Benchmark · Extracción de skills técnicas

**Fecha:** 2026-05-30 23:23  
**Modelos evaluados:** `qwen2.5-coder:7b`, `qwen3:4b`, `phi4-mini:3.8b`  
**Casos de prueba:** 8 (7 evaluados + 1 ambiguo)  

---

## Resumen global

| Modelo | Parse OK | Precision | Recall | F1 | Alucinaciones | Media/caso | Tiempo total | Score |
|--------|----------|-----------|--------|----|---------------|------------|--------------|-------|
| 🥇`qwen2.5-coder:7b` | 7/7 | 54.8% | 66.7% | 64.4% | 🟢 0 | 23.7s | 3.0 min | **0.67** |
| `qwen3:4b` | 7/7 | 69.4% | 80.6% | 71.1% | 🔴 4 | 87.9s | 11.0 min | **0.66** |
| `phi4-mini:3.8b` | 6/7 | 19.0% | 33.3% | 26.2% | 🟢 0 | 10.6s | 1.4 min | **0.34** |

> Score = 0.50×F1 + 0.20×Precision + 0.20×Recall + 0.10×(sin alucinaciones)

---

## Detalle por caso

| Caso | Tipo | Modelo | Detectadas | P | R | F1 | Alucin. | s |
|------|------|--------|------------|---|---|----|---------|---|
| C1 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 50.0% | 33.3% | 40.0% | 🟢 | 20.9 |
| C1 | true_positive | `qwen3:4b` | ✓ LangChain, LangGraph | 100.0% | 66.7% | 80.0% | 🟢 | 72.7 |
| C1 | true_positive | `phi4-mini:3.8b` | ✓ FastAPI, LangChainOpenAI, LangGraph, ToolNode | 50.0% | 66.7% | 57.1% | 🟢 | 11.3 |
| C2 | comment_trap | `qwen2.5-coder:7b` | ✓ NumPy, Pandas | 100.0% | 100.0% | 100.0% | 🟢 | 22.6 |
| C2 | comment_trap | `qwen3:4b` | ✓ LangChain, LangGraph, NumPy, OpenAI API, Pandas, RAG | 33.3% | 100.0% | 50.0% | 🔴 4 | 147.0 |
| C2 | comment_trap | `phi4-mini:3.8b` | ✓ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 8.2 |
| C3 | general | `qwen2.5-coder:7b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 26.8 |
| C3 | general | `qwen3:4b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 84.1 |
| C3 | general | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 10.9 |
| C4 | ambiguo | `qwen2.5-coder:7b` | LangChain | — | — | — | — | 17.2 |
| C4 | ambiguo | `qwen3:4b` | LangChain | — | — | — | — | 42.2 |
| C4 | ambiguo | `phi4-mini:3.8b` | ChatOpenAI, LangChainOpenAI | — | — | — | — | 9.8 |
| C5 | true_positive | `qwen2.5-coder:7b` | ✓ PyTorch, Transformers, datasets | 66.7% | 100.0% | 80.0% | 🟢 | 26.6 |
| C5 | true_positive | `qwen3:4b` | ✓ Datasets, PyTorch, Transformers | 66.7% | 100.0% | 80.0% | 🟢 | 91.0 |
| C5 | true_positive | `phi4-mini:3.8b` | ✓ Datasets, Hugging Face's Transformers, PyTorch Lightning/Datasets/Trainer, Transformers | 50.0% | 100.0% | 66.7% | 🟢 | 11.3 |
| C6 | comment_trap | `qwen2.5-coder:7b` | ✓ CSS Clamp Function, Responsive Design | 0.0% | n/a | n/a | 🟢 | 21.1 |
| C6 | comment_trap | `qwen3:4b` | ✓ ∅ | n/a | n/a | n/a | 🟢 | 33.8 |
| C6 | comment_trap | `phi4-mini:3.8b` | ✓ CSS clamp(), z-index | 0.0% | n/a | n/a | 🟢 | 9.6 |
| C7 | true_positive | `qwen2.5-coder:7b` | ✓ Anthropic, ChromaDB, SentenceTransformer | 66.7% | 66.7% | 66.7% | 🟢 | 25.9 |
| C7 | true_positive | `qwen3:4b` | ✓ Anthropic, ChromaDB, SentenceTransformers | 66.7% | 66.7% | 66.7% | 🟢 | 136.9 |
| C7 | true_positive | `phi4-mini:3.8b` | ✓ AnthropicAI, ChromaDB, SentenceTransformer | 33.3% | 33.3% | 33.3% | 🟢 | 10.9 |
| C8 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 0.0% | 0.0% | 0.0% | 🟢 | 21.7 |
| C8 | true_positive | `qwen3:4b` | ✓ CrewAI, LangChain | 50.0% | 50.0% | 50.0% | 🟢 | 50.0 |
| C8 | true_positive | `phi4-mini:3.8b` | ✓ Docker, FastAPI, LangChain, PyTorch | 0.0% | 0.0% | 0.0% | 🟢 | 11.7 |

---

## Conclusión

### Modelo recomendado: `qwen2.5-coder:7b`

Con un score de **0.67**, `qwen2.5-coder:7b` obtiene el mejor balance entre precisión (54.8%), recall (66.7%) y F1 (64.4%).

La ventaja sobre el segundo clasificado es ajustada (Δ={score_gap:.2f}) — ambos son opciones viables.

**Alucinaciones:** Solo `qwen2.5-coder:7b`, `phi4-mini:3.8b` evitó las alucinaciones.

**Latencia:** el modelo más rápido es `phi4-mini:3.8b` con 10.6s de media por llamada (1.4 min en total para los 8 casos).

### Caso ambiguo (C4)

El caso C4 (import sin uso real) revela la agresividad de cada modelo:

- `qwen2.5-coder:7b`: LangChain — ⚠️ agresivo
- `qwen3:4b`: LangChain — ⚠️ agresivo
- `phi4-mini:3.8b`: ChatOpenAI, LangChainOpenAI — ⚠️ agresivo

Un modelo conservador (∅ o pocos resultados en C4) es preferible para este sistema: reduce el ruido en el perfil técnico del usuario.

---
*Generado automáticamente por `benchmark_models.py`*
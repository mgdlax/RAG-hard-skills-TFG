# Benchmark · Extracción de skills técnicas

**Fecha:** 2026-05-31 19:58  
**Modelos evaluados:** `qwen2.5-coder:7b`, `qwen3:4b`, `phi4-mini:3.8b`, `granite-code:8b`  
**Casos de prueba:** 8 (7 evaluados + 1 ambiguo)  

---

## Resumen global

| Modelo | Parse OK | Precision | Recall | F1 | Alucinaciones | Media/caso | Tiempo total | Score |
|--------|----------|-----------|--------|----|---------------|------------|--------------|-------|
| 🥇`qwen2.5-coder:7b` | 7/7 | 52.4% | 66.7% | 62.2% | 🟢 0 | 25.9s | 3.3 min | **0.65** |
| `phi4-mini:3.8b` | 4/7 | 36.1% | 33.3% | 34.4% | 🟢 0 | 10.5s | 1.4 min | **0.41** |
| `granite-code:8b` | 7/7 | 33.8% | 55.6% | 45.3% | 🔴 2 | 31.6s | 4.0 min | **0.41** |
| `qwen3:4b` | 1/7 | 11.1% | 11.1% | 11.1% | 🟢 0 | 89.5s | 11.8 min | **0.20** |

> Score = 0.50×F1 + 0.20×Precision + 0.20×Recall + 0.10×(sin alucinaciones)

---

## Detalle por caso

| Caso | Tipo | Modelo | Detectadas | P | R | F1 | Alucin. | s |
|------|------|--------|------------|---|---|----|---------|---|
| C1 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 50.0% | 33.3% | 40.0% | 🟢 | 21.7 |
| C1 | true_positive | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 99.8 |
| C1 | true_positive | `phi4-mini:3.8b` | ✓ ChatOpenAI, LangGraph | 50.0% | 33.3% | 40.0% | 🟢 | 10.5 |
| C1 | true_positive | `granite-code:8b` | ✓ Docker, FastAPI, LangChain, PostgreSQL, PyTorch | 20.0% | 33.3% | 25.0% | 🟢 | 36.0 |
| C2 | comment_trap | `qwen2.5-coder:7b` | ✓ NumPy, Pandas | 100.0% | 100.0% | 100.0% | 🟢 | 26.6 |
| C2 | comment_trap | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 88.6 |
| C2 | comment_trap | `phi4-mini:3.8b` | ✓ NumPy, Pandas | 100.0% | 100.0% | 100.0% | 🟢 | 10.6 |
| C2 | comment_trap | `granite-code:8b` | ✓ NumPy, Pandas, Python | 66.7% | 100.0% | 80.0% | 🟢 | 27.8 |
| C3 | general | `qwen2.5-coder:7b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 27.8 |
| C3 | general | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 86.5 |
| C3 | general | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 11.7 |
| C3 | general | `granite-code:8b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 23.1 |
| C4 | ambiguo | `qwen2.5-coder:7b` | LangChain | — | — | — | — | 16.9 |
| C4 | ambiguo | `qwen3:4b` | ∅ | — | — | — | — | 78.6 |
| C4 | ambiguo | `phi4-mini:3.8b` | HumanMessage, LangChainOpenAI | — | — | — | — | 9.8 |
| C4 | ambiguo | `granite-code:8b` | ChatOpenAI, HumanMessage | — | — | — | — | 20.6 |
| C5 | true_positive | `qwen2.5-coder:7b` | ✓ Datasets, PyTorch, Trainer, Transformers | 50.0% | 100.0% | 66.7% | 🟢 | 37.9 |
| C5 | true_positive | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 84.6 |
| C5 | true_positive | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 13.2 |
| C5 | true_positive | `granite-code:8b` | ✓ Docker, Hugging Face, PyTorch, Transformers | 50.0% | 100.0% | 66.7% | 🟢 | 39.7 |
| C6 | comment_trap | `qwen2.5-coder:7b` | ✓ clamp() | 0.0% | n/a | n/a | 🟢 | 17.4 |
| C6 | comment_trap | `qwen3:4b` | ✗ ∅ | n/a | n/a | n/a | 🟢 | 92.3 |
| C6 | comment_trap | `phi4-mini:3.8b` | ✓ ∅ | n/a | n/a | n/a | 🟢 | 8.1 |
| C6 | comment_trap | `granite-code:8b` | ✓ Docker, FastAPI, PostgreSQL, PyTorch, Python | 0.0% | n/a | n/a | 🔴 2 | 40.7 |
| C7 | true_positive | `qwen2.5-coder:7b` | ✓ Anthropic, ChromaDB, SentenceTransformer | 66.7% | 66.7% | 66.7% | 🟢 | 27.1 |
| C7 | true_positive | `qwen3:4b` | ✓ Anthropic, ChromaDB, SentenceTransformers | 66.7% | 66.7% | 66.7% | 🟢 | 98.3 |
| C7 | true_positive | `phi4-mini:3.8b` | ✓ Anthropic, ChromaDB, SentenceTransformer | 66.7% | 66.7% | 66.7% | 🟢 | 11.1 |
| C7 | true_positive | `granite-code:8b` | ✓ FastAPI, PyTorch, Python | 0.0% | 0.0% | 0.0% | 🟢 | 22.5 |
| C8 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 0.0% | 0.0% | 0.0% | 🟢 | 22.8 |
| C8 | true_positive | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 76.4 |
| C8 | true_positive | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 8.0 |
| C8 | true_positive | `granite-code:8b` | ✓ Docker, FastAPI, LangChain, PostgreSQL, PyTorch | 0.0% | 0.0% | 0.0% | 🟢 | 31.3 |

---

## Conclusión

### Modelo recomendado: `qwen2.5-coder:7b`

Con un score de **0.65**, `qwen2.5-coder:7b` obtiene el mejor balance entre precisión (52.4%), recall (66.7%) y F1 (62.2%).

La ventaja sobre el segundo clasificado es clara (Δ=0.24).

**Alucinaciones:** Solo `qwen2.5-coder:7b`, `phi4-mini:3.8b`, `qwen3:4b` evitó las alucinaciones.

**Latencia:** el modelo más rápido es `phi4-mini:3.8b` con 10.5s de media por llamada (1.4 min en total para los 8 casos).

### Caso ambiguo (C4)

El caso C4 (import sin uso real) revela la agresividad de cada modelo:

- `qwen2.5-coder:7b`: LangChain — ⚠️ agresivo
- `qwen3:4b`: ∅ (conservador) — ✅ conservador
- `phi4-mini:3.8b`: HumanMessage, LangChainOpenAI — ⚠️ agresivo
- `granite-code:8b`: ChatOpenAI, HumanMessage — ⚠️ agresivo

Un modelo conservador (∅ o pocos resultados en C4) es preferible para este sistema: reduce el ruido en el perfil técnico del usuario.

---
*Generado automáticamente por `benchmark_models.py`*
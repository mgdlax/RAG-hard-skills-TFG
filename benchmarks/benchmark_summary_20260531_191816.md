# Benchmark · Extracción de skills técnicas

**Fecha:** 2026-05-31 18:54  
**Modelos evaluados:** `qwen2.5-coder:7b`, `qwen3:4b`, `phi4-mini:3.8b`, `granite-code:8b`  
**Casos de prueba:** 8 (7 evaluados + 1 ambiguo)  

---

## Resumen global

| Modelo | Parse OK | Precision | Recall | F1 | Alucinaciones | Media/caso | Tiempo total | Score |
|--------|----------|-----------|--------|----|---------------|------------|--------------|-------|
| 🥇`qwen2.5-coder:7b` | 7/7 | 54.8% | 66.7% | 64.4% | 🟢 0 | 22.5s | 2.9 min | **0.67** |
| `phi4-mini:3.8b` | 4/7 | 36.1% | 33.3% | 34.4% | 🟢 0 | 20.7s | 2.7 min | **0.41** |
| `granite-code:8b` | 7/7 | 33.8% | 55.6% | 45.3% | 🔴 2 | 43.1s | 5.5 min | **0.41** |
| `qwen3:4b` | 1/7 | 11.1% | 11.1% | 11.1% | 🟢 0 | 91.6s | 12.4 min | **0.20** |

> Score = 0.50×F1 + 0.20×Precision + 0.20×Recall + 0.10×(sin alucinaciones)

---

## Detalle por caso

| Caso | Tipo | Modelo | Detectadas | P | R | F1 | Alucin. | s |
|------|------|--------|------------|---|---|----|---------|---|
| C1 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 50.0% | 33.3% | 40.0% | 🟢 | 15.2 |
| C1 | true_positive | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 101.2 |
| C1 | true_positive | `phi4-mini:3.8b` | ✓ ChatOpenAI, LangGraph | 50.0% | 33.3% | 40.0% | 🟢 | 18.0 |
| C1 | true_positive | `granite-code:8b` | ✓ Docker, FastAPI, LangChain, PostgreSQL, PyTorch | 20.0% | 33.3% | 25.0% | 🟢 | 48.9 |
| C2 | comment_trap | `qwen2.5-coder:7b` | ✓ NumPy, Pandas | 100.0% | 100.0% | 100.0% | 🟢 | 25.3 |
| C2 | comment_trap | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 87.9 |
| C2 | comment_trap | `phi4-mini:3.8b` | ✓ NumPy, Pandas | 100.0% | 100.0% | 100.0% | 🟢 | 16.0 |
| C2 | comment_trap | `granite-code:8b` | ✓ NumPy, Pandas, Python | 66.7% | 100.0% | 80.0% | 🟢 | 32.2 |
| C3 | general | `qwen2.5-coder:7b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 24.8 |
| C3 | general | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 86.3 |
| C3 | general | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 17.0 |
| C3 | general | `granite-code:8b` | ✓ FastAPI, Pydantic, SQLAlchemy | 100.0% | 100.0% | 100.0% | 🟢 | 28.6 |
| C4 | ambiguo | `qwen2.5-coder:7b` | LangChain | — | — | — | — | 18.6 |
| C4 | ambiguo | `qwen3:4b` | ∅ | — | — | — | — | 100.4 |
| C4 | ambiguo | `phi4-mini:3.8b` | HumanMessage, LangChainOpenAI | — | — | — | — | 15.1 |
| C4 | ambiguo | `granite-code:8b` | ChatOpenAI, HumanMessage | — | — | — | — | 30.5 |
| C5 | true_positive | `qwen2.5-coder:7b` | ✓ Datasets, PyTorch, Transformers | 66.7% | 100.0% | 80.0% | 🟢 | 24.2 |
| C5 | true_positive | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 88.2 |
| C5 | true_positive | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 19.8 |
| C5 | true_positive | `granite-code:8b` | ✓ Docker, Hugging Face, PyTorch, Transformers | 50.0% | 100.0% | 66.7% | 🟢 | 52.3 |
| C6 | comment_trap | `qwen2.5-coder:7b` | ✓ clamp() | 0.0% | n/a | n/a | 🟢 | 22.5 |
| C6 | comment_trap | `qwen3:4b` | ✗ ∅ | n/a | n/a | n/a | 🟢 | 95.2 |
| C6 | comment_trap | `phi4-mini:3.8b` | ✓ ∅ | n/a | n/a | n/a | 🟢 | 22.8 |
| C6 | comment_trap | `granite-code:8b` | ✓ Docker, FastAPI, PostgreSQL, PyTorch, Python | 0.0% | n/a | n/a | 🔴 2 | 56.7 |
| C7 | true_positive | `qwen2.5-coder:7b` | ✓ Anthropic, ChromaDB, SentenceTransformer | 66.7% | 66.7% | 66.7% | 🟢 | 25.3 |
| C7 | true_positive | `qwen3:4b` | ✓ Anthropic, ChromaDB, SentenceTransformers | 66.7% | 66.7% | 66.7% | 🟢 | 101.3 |
| C7 | true_positive | `phi4-mini:3.8b` | ✓ Anthropic, ChromaDB, SentenceTransformer | 66.7% | 66.7% | 66.7% | 🟢 | 26.2 |
| C7 | true_positive | `granite-code:8b` | ✓ FastAPI, PyTorch, Python | 0.0% | 0.0% | 0.0% | 🟢 | 44.1 |
| C8 | true_positive | `qwen2.5-coder:7b` | ✓ ChatOpenAI, LangChain | 0.0% | 0.0% | 0.0% | 🟢 | 20.0 |
| C8 | true_positive | `qwen3:4b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 81.3 |
| C8 | true_positive | `phi4-mini:3.8b` | ✗ ∅ | 0.0% | 0.0% | 0.0% | 🟢 | 25.2 |
| C8 | true_positive | `granite-code:8b` | ✓ Docker, FastAPI, LangChain, PostgreSQL, PyTorch | 0.0% | 0.0% | 0.0% | 🟢 | 38.6 |

---

## Conclusión

### Modelo recomendado: `qwen2.5-coder:7b`

Con un score de **0.67**, `qwen2.5-coder:7b` obtiene el mejor balance entre precisión (54.8%), recall (66.7%) y F1 (64.4%).

La ventaja sobre el segundo clasificado es clara (Δ=0.25).

**Alucinaciones:** Solo `qwen2.5-coder:7b`, `phi4-mini:3.8b`, `qwen3:4b` evitó las alucinaciones.

**Latencia:** el modelo más rápido es `phi4-mini:3.8b` con 20.7s de media por llamada (2.7 min en total para los 8 casos).

### Caso ambiguo (C4)

El caso C4 (import sin uso real) revela la agresividad de cada modelo:

- `qwen2.5-coder:7b`: LangChain — ⚠️ agresivo
- `qwen3:4b`: ∅ (conservador) — ✅ conservador
- `phi4-mini:3.8b`: HumanMessage, LangChainOpenAI — ⚠️ agresivo
- `granite-code:8b`: ChatOpenAI, HumanMessage — ⚠️ agresivo

Un modelo conservador (∅ o pocos resultados en C4) es preferible para este sistema: reduce el ruido en el perfil técnico del usuario.

---
*Generado automáticamente por `benchmark_models.py`*
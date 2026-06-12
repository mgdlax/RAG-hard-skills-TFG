# Benchmark · Modelos de Embeddings

**Fecha:** 2026-06-02 00:04  
**Modelos:** all-MiniLM-L6-v2 vs multilingual-e5-small  
**Casos:** 10  

---

## Resumen global

| Modelo | Hit@1 | Avg Gap | Avg Sim Rel | Avg Sim Irr | ms/enc | Score |
|--------|-------|---------|-------------|-------------|--------|-------|
| 🥇 `all-MiniLM-L6-v2` | 90.0% | +0.1707 | 0.3399 | 0.1692 | 66.5ms | **0.7740** |
| `multilingual-e5-small` | 90.0% | +0.0406 | 0.8332 | 0.7927 | 168.8ms | **0.6775** |

> Score = 0.5×Hit@1 + 0.3×norm(AvgGap) + 0.2×AvgSimRel

---

## Detalle por caso

| Caso | Tipo | Modelo | SimRel | SimIrr | Gap | Hit@1 | ms/enc |
|------|------|--------|--------|--------|-----|-------|--------|
| C01 | cross_lingual | `all-MiniLM-L6-v2` | 0.4279 | 0.1400 | +0.2878 | ✓ | 89.4ms |
| C01 | cross_lingual | `multilingual-e5-small` | 0.8229 | 0.7686 | +0.0543 | ✓ | 94.6ms |
| C02 | exact_skill | `all-MiniLM-L6-v2` | 0.4218 | 0.0065 | +0.4153 | ✓ | 28.9ms |
| C02 | exact_skill | `multilingual-e5-small` | 0.8552 | 0.7882 | +0.0670 | ✓ | 68.7ms |
| C03 | inference | `all-MiniLM-L6-v2` | 0.3231 | 0.2401 | +0.0830 | ✓ | 31.5ms |
| C03 | inference | `multilingual-e5-small` | 0.8180 | 0.7662 | +0.0518 | ✓ | 90.0ms |
| C04 | false_positive_trap | `all-MiniLM-L6-v2` | 0.6575 | 0.2300 | +0.4275 | ✓ | 26.0ms |
| C04 | false_positive_trap | `multilingual-e5-small` | 0.8892 | 0.8143 | +0.0749 | ✓ | 140.0ms |
| C05 | domain_trap | `all-MiniLM-L6-v2` | 0.2411 | 0.0885 | +0.1525 | ✓ | 31.3ms |
| C05 | domain_trap | `multilingual-e5-small` | 0.7991 | 0.7869 | +0.0122 | ✓ | 178.0ms |
| C06 | cross_lingual | `all-MiniLM-L6-v2` | 0.2735 | 0.2043 | +0.0692 | ✓ | 24.6ms |
| C06 | cross_lingual | `multilingual-e5-small` | 0.8379 | 0.8172 | +0.0206 | ✓ | 140.0ms |
| C07 | exact_skill | `all-MiniLM-L6-v2` | 0.1199 | -0.0479 | +0.1678 | ✓ | 138.6ms |
| C07 | exact_skill | `multilingual-e5-small` | 0.8088 | 0.7531 | +0.0556 | ✓ | 358.1ms |
| C08 | exact_skill | `all-MiniLM-L6-v2` | 0.3260 | 0.2393 | +0.0867 | ✓ | 77.8ms |
| C08 | exact_skill | `multilingual-e5-small` | 0.8347 | 0.8204 | +0.0144 | ✓ | 139.0ms |
| C09 | inference | `all-MiniLM-L6-v2` | 0.1202 | 0.3518 | -0.2316 | ✗ | 129.4ms |
| C09 | inference | `multilingual-e5-small` | 0.8059 | 0.8085 | -0.0026 | ✗ | 309.1ms |
| C10 | cross_lingual | `all-MiniLM-L6-v2` | 0.4876 | 0.2390 | +0.2486 | ✓ | 88.0ms |
| C10 | cross_lingual | `multilingual-e5-small` | 0.8608 | 0.8033 | +0.0574 | ✓ | 170.1ms |

---

## Análisis por tipo de caso

| Tipo | Modelo | Hit@1 | Avg Gap |
|------|--------|-------|---------|
| cross_lingual | `all-MiniLM-L6-v2` | 100% | +0.2019 |
| cross_lingual | `multilingual-e5-small` | 100% | +0.0441 |
| domain_trap | `all-MiniLM-L6-v2` | 100% | +0.1525 |
| domain_trap | `multilingual-e5-small` | 100% | +0.0122 |
| exact_skill | `all-MiniLM-L6-v2` | 100% | +0.2233 |
| exact_skill | `multilingual-e5-small` | 100% | +0.0456 |
| false_positive_trap | `all-MiniLM-L6-v2` | 100% | +0.4275 |
| false_positive_trap | `multilingual-e5-small` | 100% | +0.0749 |
| inference | `all-MiniLM-L6-v2` | 50% | -0.0743 |
| inference | `multilingual-e5-small` | 50% | +0.0246 |

---

## Conclusión

### Modelo recomendado: `all-MiniLM-L6-v2`

Score **0.7740** vs 0.6775 del rival (Δ=0.0965). La ventaja es clara.

- **Hit@1 rate**: 90.0% — el modelo discrimina correctamente entre documento relevante e irrelevante en 9/10 casos.
- **Gap medio**: +0.1707 — separación media entre similitud correcta e incorrecta.
- **Latencia**: 66.5ms por texto.

### Nota sobre `multilingual-e5-small`

Este modelo requiere los prefijos `query:` y `passage:` según su documentación oficial. Sin ellos, su rendimiento puede degradarse significativamente en tareas de recuperación.

---
*Generado automáticamente por `eval_embeddings.py`*
# RAG GitHub Skills

> Sistema de recomendación técnica de desarrolladores basado en evidencias extraídas de repositorios públicos de GitHub mediante Recuperación Aumentada por Generación (RAG).

**TFG · Grado en Ingeniería del Software**

---

## Índice

1. [Visión general](#1-visión-general)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Requisitos previos](#3-requisitos-previos)
4. [Instalación](#4-instalación)
5. [Configuración](#5-configuración)
6. [Interfaz web (Streamlit)](#6-interfaz-web-streamlit)
7. [Pipeline por línea de comandos](#7-pipeline-por-línea-de-comandos)
8. [Sistema RAG: consultas](#8-sistema-rag-consultas)
9. [Comprobar el estado del sistema](#9-comprobar-el-estado-del-sistema)
10. [Evaluación del sistema](#10-evaluación-del-sistema)
11. [Descripción de módulos](#11-descripción-de-módulos)
12. [Formatos de datos](#12-formatos-de-datos)
13. [Benchmarking](#13-benchmarking)
14. [Estructura de directorios](#14-estructura-de-directorios)
15. [Solución de problemas](#15-solución-de-problemas)
16. [Actualización de perfiles bajo demanda](#16-actualización-de-perfiles-bajo-demanda)

---

## 1. Visión general

**RAG GitHub Skills** analiza los repositorios públicos de un desarrollador en GitHub y construye un perfil técnico basado en evidencias reales: ficheros de código, commits, issues y pull requests. El objetivo es ayudar a equipos de RRHH técnico a identificar candidatos cualificados con trazabilidad completa a las fuentes.

### Flujo de alto nivel

```
GitHub (repos públicos)
        |
  FASE 1  Ingesta            ficheros, commits, issues, PRs -> JSONL crudo
        |
  FASE 2  Procesamiento      métricas de calidad + LLM detecta skills -> JSONL procesado
        |
  FASE 3  Perfil técnico     agrega skills por usuario -> JSON de perfil
        |
  FASE 4  Indexación         embeddings + ChromaDB -> índice vectorial
        |
  FASE 5  Retrieval+Ranking  consulta semántica + scoring multi-usuario
        |
  FASE 6  Generación         Ollama sintetiza la respuesta con citas [N]
```

### Tecnologías clave

| Componente | Tecnología |
|------------|-----------|
| API de GitHub | PyGithub |
| Detección de skills y generación | Ollama + `qwen2.5-coder:7b` |
| Embeddings | sentence-transformers, `isuruwijesiri/all-MiniLM-L6-v2-code-search-512` (384 dims) |
| Base de datos vectorial | ChromaDB (HNSW, distancia coseno) |
| Validación de datos | Pydantic v2 |
| Interfaz web | Streamlit |

---

## 2. Arquitectura del sistema

```
+------------------------------------------------------------------+
|                          INTERFAZ WEB                            |
|  app.py  ·  frontend/ (componentes)  ·  services/ (lógica)       |
+--------------------------------+---------------------------------+
                                 |
+--------------------------------v---------------------------------+
|                          BACKEND (src/)                          |
|                                                                  |
|  ingestion/      FASE 1   cliente GitHub, extracción de repos    |
|  processing/     FASE 2   métricas estructurales + LLM           |
|  profiling/      FASE 3   perfil técnico agregado por skill      |
|  vectorization/  FASE 4   embeddings + colección ChromaDB        |
|  rag/            FASES 5-6 retrieval, ranking, generación        |
|  evaluation/     métricas IR: P@k, Recall@k, MRR, NDCG@k         |
|  utils/          logging dual (consola + fichero)                |
|  config.py       constantes globales del sistema                 |
+------------------------------------------------------------------+
```

---

## 3. Requisitos previos

| Software | Versión mínima | Para qué se usa |
|----------|---------------|-----------------|
| Python | 3.10+ | Runtime del sistema |
| Ollama | Última | LLM local para detección de skills y generación |

**Token de GitHub:** se necesita un Personal Access Token con permiso de lectura de repos públicos (GitHub → Settings → Developer settings → Personal access tokens).

**Modelo LLM local:**

```bash
ollama pull qwen2.5-coder:7b
```

El sistema funciona con cualquier modelo compatible con Ollama; cambiar `DEFAULT_LLM_MODEL` en `src/config.py`.

---

## 4. Instalación

```bash
# 1. Situarse en la raíz del proyecto
cd tfg-rag-github

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 5. Configuración

### 5.1 Variables de entorno

Crear el fichero `.env` en la raíz del proyecto:

```env
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

El token solo necesita acceso de lectura a repos públicos.

### 5.2 Constantes del sistema (`src/config.py`)

```python
OLLAMA_URL        = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"
EMBEDDING_MODEL   = "isuruwijesiri/all-MiniLM-L6-v2-code-search-512"
CHROMA_PATH       = "data/chromadb"
CHROMA_COLLECTION = "technical_blocks"
```

**Importante:** si se cambia `EMBEDDING_MODEL`, hay que borrar `data/chromadb/` y volver a indexar todos los usuarios: los vectores almacenados son incompatibles entre modelos.

### 5.3 Modo mock del frontend

Por defecto la interfaz usa el pipeline real. Para desarrollar el frontend sin GitHub, Ollama ni ChromaDB, activar el modo simulado en ambos servicios:

```python
# services/profile_service.py y services/rag_service.py
USE_MOCK: bool = True
```

---

## 6. Interfaz web (Streamlit)

### Arranque

```bash
streamlit run app.py
```

Se abre en `http://localhost:8501`.

### Comportamiento al arrancar

Al iniciar, la aplicación **carga automáticamente los perfiles ya indexados en ChromaDB** (no hace falta re-procesar nada). Si hay perfiles, el chat queda habilitado de inmediato.

### Añadir un perfil nuevo

1. Escribir el username de GitHub en la barra lateral.
2. Pulsar "Añadir perfil". Se ejecuta el pipeline completo (fases 1-4), lo que puede tardar varios minutos por usuario.
3. Al terminar, el perfil aparece en la lista con sus skills detectadas.

### Hacer una consulta

Escribir la pregunta en el chat. La respuesta incluye:

- **Texto narrativo** generado por el LLM con citas `[N]` a las evidencias.
- **Ranking de candidatos** con barra de score y skills coincidentes.
- **Tarjetas de evidencias** con repo, fichero, explicación y fragmento de código.

Si Ollama no está disponible, el sistema devuelve una respuesta estructurada con los mismos datos (degradación elegante), nunca un error.

---

## 7. Pipeline por línea de comandos

### Indexar usuarios (fases 1-4)

```bash
python -m src.main                  # primer usuario de TARGET_USERS
python -m src.main chitralputhran   # un usuario concreto
python -m src.main --all            # los 5 usuarios objetivo del TFG
```

Los usuarios objetivo están definidos en `TARGET_USERS` (`src/main.py`) y coinciden con el ground truth de evaluación:

| Usuario | Perfil técnico |
|---------|----------------|
| chitralputhran | RAG + LangGraph + Streamlit, apps de IA generativa |
| ranguy9304 | RAG terminal con LangGraph, sistemas distribuidos |
| naurjhanvi | Edge AI + RAG + LSTM, despliegue Docker/Kubernetes/FastAPI |
| yldzburhan | ML/data science, NLP, sistemas de recomendación |
| HalemoGPA | Deep learning, PyTorch, visión por computador, NLP |

### Parámetros del pipeline (en `run_pipeline`, `src/main.py`)

| Parámetro | Por defecto | Descripción |
|-----------|------------|-------------|
| `max_repos` | 6 | Repositorios más relevantes a analizar |
| `max_files_per_repo` | 50 | Ficheros máximos por repositorio |
| `max_commits_per_repo` | 20 | Commits máximos por repositorio |
| `max_issues_per_repo` | 10 | Issues máximas por repositorio |
| `max_prs_per_repo` | 10 | Pull requests máximas por repositorio |
| `limit_files` | 30 | Ficheros de código enviados al LLM |

La salida de cada fase queda registrada en consola y, con nivel DEBUG, en `logs/pipeline_<usuario>_<timestamp>.log`.

---

## 8. Sistema RAG: consultas

### Consulta única

```bash
python -m src.rag.query "quien recomiendas para construir sistemas RAG?"
```

### Modo interactivo

```bash
python -m src.rag.query
```

Muestra los usuarios indexados y permite encadenar consultas. Escribir `salir` para terminar.

### Funcionamiento interno de una consulta

```
consulta en lenguaje natural
   | embed_texts()                  mismo modelo que la indexación
   | ChromaDB query                 similitud coseno sobre HNSW
30 bloques candidatos (máx. 10 por usuario, similitud >= 0.55)
   | rank_candidates()
   |   combined_score = similitud * 0.40 + composite_score * 0.60
   |   por skill: mejor bloque; por usuario: media + bonus de amplitud
   | generate_response()            prompt con evidencias numeradas [N]
respuesta narrativa + sección de referencias determinista
   | evaluate_answer_against_profiles()
consistencia de la respuesta frente a los perfiles técnicos
```

### Ejemplos de consultas

```bash
python -m src.rag.query "quien tiene mas experiencia con LangGraph?"
python -m src.rag.query "quien domina deep learning y vision por computador con PyTorch?"
python -m src.rag.query "quien sabe desplegar modelos ML con Docker y Kubernetes?"
python -m src.rag.query "quien recomiendas para un sistema de recomendacion y analisis de datos?"
```

---

## 9. Comprobar el estado del sistema

### Estado de ChromaDB (usuarios y bloques indexados)

```bash
python -c "from src.vectorization.indexer import get_index_stats; import json; print(json.dumps(get_index_stats(), indent=2))"
```

Salida esperada: total de bloques, bloques por usuario y skills únicas por usuario.

### Estado de Ollama

```bash
curl http://localhost:11434/api/tags     # lista los modelos disponibles
ollama list                              # equivalente desde el CLI de Ollama
```

### Perfiles generados

```bash
ls data/perfiles/        # un <username>_profile.json por usuario indexado
```

### Comprobación rápida de extremo a extremo

```bash
python -m src.rag.query "quien sabe de LangChain?"
```

Si ChromaDB está vacío lo indica; si Ollama está caído devuelve la respuesta estructurada de fallback. En ambos casos el comando termina sin errores.

---

## 10. Evaluación del sistema

### Ejecutar la evaluación

```bash
python -m src.evaluation.evaluator           # ajusta a los usuarios disponibles
python -m src.evaluation.evaluator --strict  # solo consultas con todos sus usuarios indexados
```

El informe completo se guarda en `data/evaluation/report_<timestamp>.json`.

### Métricas calculadas

| Métrica | Descripción |
|---------|-------------|
| Precision@1 | Fracción de consultas cuyo primer resultado es correcto |
| Precision@3 | Fracción de correctos entre los 3 primeros |
| Recall@3 | Fracción de usuarios relevantes encontrados en el top-3 |
| MRR | Reciprocal rank medio del primer resultado correcto |
| NDCG@3 | Ganancia acumulada normalizada con relevancia graduada (0-3) |

Interpretación orientativa: MRR >= 0.80 y P@1 >= 0.70 es excelente; MRR >= 0.60 y P@1 >= 0.50 es un resultado sólido.

### Ground truth

`src/evaluation/ground_truth.py` define 10 consultas anotadas manualmente sobre los 5 usuarios objetivo, agrupadas en 6 categorías: `rag_systems`, `orchestration`, `deployment`, `deep_learning`, `nlp` y `data_science`. Cada consulta incluye los usuarios relevantes (relevancia binaria) y grados 0-3 para NDCG.

### Resultados de referencia (última ejecución)

```
Consultas evaluadas : 10
Precision@1         : 0.8000
Precision@3         : 0.5667
Recall@3            : 0.8333
MRR                 : 0.8833
NDCG@3              : 0.7316
```

---

## 11. Descripción de módulos

### Backend (`src/`)

| Módulo | Responsabilidad | Ejecución directa |
|--------|-----------------|-------------------|
| `config.py` | Constantes globales: URL de Ollama, modelos, rutas de ChromaDB | No (se importa) |
| `main.py` | Orquesta las fases 1-4 del pipeline para un usuario | `python -m src.main [user\|--all]` |
| `ingestion/github_client.py` | Cliente PyGithub autenticado con el token del `.env` | No |
| `ingestion/schemas.py` | Modelo Pydantic `GitHubEvidence` (contrato de la fase 1) | No |
| `ingestion/extract_repo.py` | Descarga ficheros, commits, issues y PRs de un repo; filtra extensiones relevantes y convierte notebooks a código plano | No |
| `ingestion/extract_user.py` | Selecciona los mejores repos del usuario (stars + actividad, sin forks ni archivados) y delega en `extract_repo` | No |
| `processing/metrics.py` | 4 métricas estructurales sin LLM: recency (15%), authorship (20%), artifact_weight (30%), content_richness (35%) → `composite_score` | No |
| `processing/prompts.py` | Prompt estricto de detección de skills y modelos Pydantic de la respuesta del LLM | No |
| `processing/code_filter.py` | Por cada evidencia: métricas + llamada a Ollama + parseo robusto del JSON; emite un bloque por skill | No |
| `profiling/build_profile.py` | Agrega los bloques en un perfil por skill con confianza (alta/media/baja) y trazabilidad | No |
| `vectorization/embedder.py` | Singleton del modelo de embeddings; normalización L2 y diagnóstico de longitud en tokens | No |
| `vectorization/indexer.py` | Construye los documentos a vectorizar (skill + explicación + fragmento), indexa con upsert y expone `get_index_stats()` | No |
| `rag/retrieval.py` | Embedding de la consulta + búsqueda en ChromaDB con filtros de similitud y límite por usuario | No |
| `rag/ranking.py` | Agrupa bloques por usuario, calcula `final_score` y asigna confianza al ranking | No |
| `rag/generation.py` | Prompt RAG con evidencias numeradas, llamada a Ollama y referencias deterministas; fallback sin LLM | No |
| `rag/answer_evaluation.py` | Contrasta la respuesta generada con los perfiles técnicos (consistencia, skills sin soporte, sobre-afirmaciones) | No |
| `rag/query.py` | Punto de entrada del RAG: consulta única o modo interactivo | `python -m src.rag.query ["consulta"]` |
| `evaluation/ground_truth.py` | 10 consultas anotadas con relevancia binaria y graduada | No |
| `evaluation/metrics.py` | Funciones puras de métricas IR: P@k, R@k, RR, NDCG@k | No |
| `evaluation/evaluator.py` | Evalúa el pipeline sobre el ground truth y genera el informe JSON | `python -m src.evaluation.evaluator [--strict]` |
| `utils/logger.py` | Logging dual: consola con colores (INFO+) y fichero (DEBUG+) en `logs/` | No |

### Frontend

| Módulo | Responsabilidad |
|--------|-----------------|
| `app.py` | Punto de entrada Streamlit: estado de sesión, precarga de perfiles indexados, barra lateral y chat | `streamlit run app.py` |
| `frontend/layout.py` | Cabecera principal y bloques de la barra lateral (título, métricas del sistema) |
| `frontend/components.py` | Tarjetas de perfil, ranking con barras de score, tarjetas de evidencias, pantalla de bienvenida. Todo el contenido dinámico se escapa antes de inyectarse en HTML |
| `frontend/styles.py` | CSS personalizado inyectado en la app (tema oscuro) |
| `services/profile_service.py` | `add_profile()` ejecuta el pipeline real (o mock) emitiendo pasos de progreso; `load_indexed_profiles()` reconstruye los perfiles ya indexados al arrancar |
| `services/rag_service.py` | `ask_question()` ejecuta retrieval + ranking + generación (o mock) y adapta el resultado al formato de la UI |

---

## 12. Formatos de datos

### JSONL crudo — `data/raw/<username>.jsonl`

Un objeto `GitHubEvidence` por línea:

```json
{
  "username": "chitralputhran",
  "repo": "chitralputhran/Advanced-RAG-LangGraph",
  "artifact_type": "file",
  "path": "document_loader.py",
  "author": null,
  "text": "from langchain...",
  "metadata": {
    "source": "github",
    "url": "https://github.com/...",
    "size": 8432,
    "repo_pushed_at": "2026-01-15T10:23:41"
  }
}
```

### JSONL procesado — `data/procesado/<username>_processed.jsonl`

Un bloque por skill detectada (una evidencia puede generar N bloques):

```json
{
  "username": "chitralputhran",
  "skill": "LangChain",
  "source": {
    "repo": "chitralputhran/Advanced-RAG-LangGraph",
    "path": "document_loader.py",
    "artifact_type": "file",
    "url": "https://github.com/..."
  },
  "scores": {
    "recency": 0.91,
    "authorship": 0.9,
    "artifact_weight": 0.82,
    "content_richness": 0.9,
    "composite_score": 0.87
  },
  "explanation": "Carga de documentos con loaders de LangChain.",
  "evidence_fragment": "from langchain_community.document_loaders import ..."
}
```

### JSON de perfil — `data/perfiles/<username>_profile.json`

```json
{
  "user": "chitralputhran",
  "profile_source": "data/procesado/chitralputhran_processed.jsonl",
  "total_skill_blocks": 233,
  "unique_skills": 43,
  "technical_profile": {
    "hard_skills": [
      {
        "skill": "LangChain",
        "confidence": "alta",
        "evidence_count": 23,
        "avg_composite_score": 0.83,
        "evidence_diversity_score": 0.5,
        "evidence_repositories": ["chitralputhran/Advanced-RAG-LangGraph"],
        "evidence_files": ["document_loader.py"],
        "artifact_types": ["commit", "file"],
        "sample_explanations": ["..."],
        "sample_signals": ["..."]
      }
    ]
  }
}
```

---

## 13. Benchmarking

Scripts independientes usados para justificar la elección de modelos del TFG:

```bash
# Comparativa de modelos LLM locales en detección de skills
# (precision, recall, F1, alucinaciones, trampas de comentarios)
python benchmark_models.py

# Comparativa de modelos de embeddings para el retrieval
python eval_embeddings.py
```

Ambos requieren tener descargados en Ollama los modelos listados en la constante `MODELS` de cada script. Los resultados (CSV + resumen Markdown) se generan en la raíz; los históricos de ejecuciones anteriores están archivados en `benchmarks/`.

---

## 14. Estructura de directorios

```
tfg-rag-github/
├── app.py                      # Punto de entrada de la interfaz Streamlit
├── requirements.txt            # Dependencias Python
├── .env                        # Token de GitHub (no versionar)
│
├── src/                        # Backend
│   ├── config.py               # Constantes globales
│   ├── main.py                 # Pipeline CLI (fases 1-4)
│   ├── ingestion/              # FASE 1: extracción desde GitHub
│   ├── processing/             # FASE 2: métricas + detección LLM
│   ├── profiling/              # FASE 3: perfil técnico agregado
│   ├── vectorization/          # FASE 4: embeddings + ChromaDB
│   ├── rag/                    # FASES 5-6: retrieval, ranking, generación
│   ├── evaluation/             # Evaluación IR contra ground truth
│   └── utils/                  # Logging
│
├── frontend/                   # Componentes visuales de la UI
├── services/                   # Lógica de negocio de la UI (real/mock)
│
├── data/
│   ├── raw/                    # JSONL crudo por usuario
│   ├── procesado/              # JSONL de bloques por skill
│   ├── perfiles/               # JSON de perfil técnico por usuario
│   ├── chromadb/               # Índice vectorial persistente
│   └── evaluation/             # Informes de evaluación JSON
│
├── benchmarks/                 # Resultados históricos de benchmarking
├── benchmark_models.py         # Benchmark de modelos LLM
├── eval_embeddings.py          # Benchmark de modelos de embeddings
└── logs/                       # Logs del pipeline con timestamp
```

---

## 15. Solución de problemas

### "No se encontró GITHUB_TOKEN"

Verificar que existe `.env` en la raíz con `GITHUB_TOKEN=ghp_...`.

### Ollama no responde

```bash
curl http://localhost:11434/api/tags   # comprobar que está activo
ollama serve                           # arrancarlo si no lo está
ollama pull qwen2.5-coder:7b           # descargar el modelo si falta
```

El sistema sigue funcionando sin Ollama: la fase 2 del pipeline no podrá detectar skills, pero las consultas RAG devuelven una respuesta estructurada sin narrativa.

### "ChromaDB está vacío"

Indexar al menos un usuario antes de consultar:

```bash
python -m src.main chitralputhran
```

### La fase 2 tarda mucho

Se llama al LLM una vez por evidencia. Para pruebas rápidas, reducir `limit_files` en `src/main.py` (por ejemplo de 30 a 10).

### Eliminar un usuario del índice

```python
from src.vectorization.indexer import get_collection
get_collection().delete(where={"username": "nombre_usuario"})
```

Para re-indexarlo después: `python -m src.main nombre_usuario` (el `upsert` evita duplicados).

### Los logs del modo --all se mezclan

En modo `--all` todos los usuarios comparten el fichero de log del primero (el logger se configura una sola vez). Para logs separados, indexar cada usuario en un comando independiente.

---

---

## 16. Actualización de perfiles bajo demanda

Los perfiles indexados pueden quedarse desactualizados cuando el usuario sube nuevos commits o repositorios a GitHub. Para refrescarlos hay un botón **"Buscar cambios y actualizar"** en la barra lateral del frontend, que **solo re-indexa los usuarios con actividad real en GitHub**. Así se evita lanzar el pipeline —que puede tardar varios minutos— cuando no hay nada nuevo.

### Cómo funciona

```
Botón "Buscar cambios y actualizar"  (o  python update_profiles.py)
  └─ get_indexed_usernames()           lista de usuarios en ChromaDB
       └─ para cada usuario:
            ¿algún repo.pushed_at > last_run_at guardado en el perfil JSON?
              ├─ No → skip (sin cambios, sin coste)
              └─ Sí → run_pipeline(username)
                         ├─ Fase 1: ingesta desde GitHub
                         ├─ Fase 2: métricas + detección de skills con LLM
                         ├─ Fase 3: perfil técnico (actualiza last_run_at)
                         └─ Fase 4: upsert en ChromaDB (sin duplicados)
```

La detección de cambios compara el campo `pushed_at` de cada repositorio del usuario —disponible en la API de GitHub sin coste adicional de rate limit— contra el campo `last_run_at` guardado en `data/perfiles/<username>_profile.json` al final de cada ejecución del pipeline.

### Lógica reutilizable — `services/profile_service.py`

Toda la lógica vive en la función `update_profiles()`, un generador que va informando del progreso (mensaje, is_final, resumen). La usan tanto el botón del frontend ([`app.py`](app.py)) como el script de consola, de modo que el comportamiento es exactamente el mismo en ambos casos.

### Desde la línea de comandos

```bash
python update_profiles.py
```

Muestra qué usuarios se actualizan y cuáles se saltan. Para forzar la actualización de todos, borrar el campo `last_run_at` de un perfil JSON o eliminar el fichero.

---

*TFG · Ingeniería del Software — Sistema de recomendación técnica basado en evidencias de GitHub*

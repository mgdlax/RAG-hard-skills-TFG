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
6. [Modos de ejecución](#6-modos-de-ejecución)
7. [Interfaz web — Streamlit](#7-interfaz-web--streamlit)
8. [Pipeline por línea de comandos](#8-pipeline-por-línea-de-comandos)
9. [Sistema RAG — Consultas](#9-sistema-rag--consultas)
10. [Evaluación del sistema](#10-evaluación-del-sistema)
11. [Descripción de módulos](#11-descripción-de-módulos)
12. [Formatos de datos](#12-formatos-de-datos)
13. [Benchmarking](#13-benchmarking)
14. [Ejemplos de consultas](#14-ejemplos-de-consultas)
15. [Estructura de directorios](#15-estructura-de-directorios)
16. [Solución de problemas](#16-solución-de-problemas)

---

## 1. Visión general

**RAG GitHub Skills** analiza los repositorios públicos de un desarrollador en GitHub y construye un perfil técnico basado en evidencias reales: ficheros de código, commits, issues y pull requests. El objetivo es ayudar a equipos de RRHH técnico a identificar candidatos cualificados con trazabilidad completa a las fuentes.

### ¿Qué problema resuelve?

Las entrevistas técnicas y los CVs son subjetivos. Este sistema extrae evidencias objetivas y verificables directamente del código que el desarrollador ha escrito y publicado.

### Flujo de alto nivel

```
GitHub (repos públicos)
        │
        ▼
  FASE 1 · Ingesta ──────────── Ficheros, commits, issues, PRs → JSONL crudo
        │
        ▼
  FASE 2 · Procesamiento ─────── Métricas de calidad + LLM detecta skills → JSONL procesado
        │
        ▼
  FASE 3 · Perfil técnico ────── Agrega skills por usuario → JSON de perfil
        │
        ▼
  FASE 4 · Indexación ────────── Embeddings + ChromaDB → Índice vectorial
        │
        ▼
  FASE 5 · Retrieval + Ranking ── Consulta semántica + scoring multi-usuario
        │
        ▼
  FASE 6 · Generación ───────── Ollama sintetiza respuesta con citas [N]
```

### Tecnologías clave

| Componente | Tecnología |
|------------|-----------|
| API de GitHub | PyGithub |
| Detección de skills | Ollama + qwen2.5-coder:7b |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Base de datos vectorial | ChromaDB (HNSW coseno) |
| Validación de datos | Pydantic v2 |
| Interfaz web | Streamlit |
| Visualización | Plotly + Pandas |

---

## 2. Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERFAZ WEB                             │
│   app.py · frontend/ · services/                                │
│   Streamlit — chat RAG con historial, ranking y evidencias      │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       BACKEND (src/)                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  ingestion/  │  │ processing/  │  │     profiling/       │  │
│  │              │  │              │  │                      │  │
│  │ github_client│  │ metrics.py   │  │ build_profile.py     │  │
│  │ extract_user │  │ code_filter  │  │                      │  │
│  │ extract_repo │  │ prompts.py   │  │ Agrega skills por    │  │
│  │ schemas.py   │  │              │  │ usuario con          │  │
│  │              │  │ Score + LLM  │  │ confianza y          │  │
│  │ FASE 1       │  │ FASE 2       │  │ trazabilidad         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                     │              │
│  ┌──────▼─────────────────▼─────────────────────▼───────────┐  │
│  │                   vectorization/                          │  │
│  │          embedder.py · indexer.py · ChromaDB              │  │
│  │          FASE 4: all-MiniLM-L6-v2 + HNSW coseno          │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                        rag/                               │  │
│  │  retrieval.py · ranking.py · generation.py · query.py     │  │
│  │  FASES 5-6: embed query → ChromaDB → rank → Ollama        │  │
│  └──────────────────────────┬────────────────────────────────┘  │
│                             │                                   │
│  ┌──────────────────────────▼────────────────────────────────┐  │
│  │                    evaluation/                            │  │
│  │  evaluator.py · metrics.py · ground_truth.py              │  │
│  │  P@k · Recall@k · MRR · NDCG@k                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  utils/logger.py · config.py                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Requisitos previos

### Software necesario

| Software | Versión mínima | Para qué se usa |
|----------|---------------|-----------------|
| Python | 3.10+ | Runtime del sistema |
| Ollama | Última | LLM local para detección de skills y generación |
| Git | Cualquiera | Clonar el repositorio |

### Cuenta de GitHub

Se necesita un **Personal Access Token (PAT)** de GitHub con permiso `public_repo` (solo lectura de repos públicos). Se puede crear en: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic).

### Modelo de lenguaje local

Instalar Ollama y descargar el modelo por defecto:

```bash
# Instalar Ollama: https://ollama.ai
ollama pull qwen2.5-coder:7b
```

> El sistema también funciona con `llama3.2`, `mistral`, `codellama` u otros modelos compatibles con Ollama. Cambiar `DEFAULT_LLM_MODEL` en `src/config.py`.

---

## 4. Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd tfg-rag-github

# 2. Crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

### Dependencias instaladas

```
PyGithub>=2.1.1          # Cliente de la API REST de GitHub
python-dotenv>=1.0.0     # Carga de variables de entorno desde .env
pydantic>=2.5.0          # Validación de datos (schemas del pipeline)
tqdm>=4.66.0             # Barras de progreso en CLI
requests>=2.31.0         # Llamadas HTTP a Ollama
sentence-transformers>=2.7.0  # Modelo de embeddings all-MiniLM-L6-v2
chromadb>=0.5.0          # Base de datos vectorial local
streamlit>=1.35.0        # Interfaz web
plotly>=5.20.0           # Gráficos interactivos
pandas>=2.1.0            # Tablas de datos
```

---

## 5. Configuración

### 5.1 Variables de entorno

Crear el fichero `.env` en la raíz del proyecto:

```env
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> El token solo necesita acceso de lectura a repos públicos. No dar permisos de escritura.

### 5.2 Constantes del sistema (`src/config.py`)

```python
OLLAMA_URL        = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"   # Cambiar aquí para otro modelo
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"   # No cambiar sin re-indexar todo
CHROMA_PATH       = "data/chromadb"       # Directorio de la base vectorial
CHROMA_COLLECTION = "technical_blocks"    # Nombre de la colección en ChromaDB
```

> **Importante:** si se cambia `EMBEDDING_MODEL`, hay que borrar `data/chromadb/` y volver a indexar todos los usuarios, porque los vectores almacenados son incompatibles con el nuevo modelo.

### 5.3 Modos de la interfaz web

Los servicios del frontend tienen dos modos controlados por flags en `services/`:

| Flag | Fichero | Efecto |
|------|---------|--------|
| `USE_MOCK = True` | `services/profile_service.py` | Simula el pipeline sin GitHub ni Ollama |
| `USE_MOCK = True` | `services/rag_service.py` | Simula el RAG con respuestas ficticias plausibles |
| `USE_MOCK = False` | Ambos | Usa el pipeline real completo |

Para una demo rápida sin configurar nada: dejar `USE_MOCK = True` (valor por defecto).

---

## 6. Modos de ejecución

El sistema tiene dos modos de uso: **interfaz web** (Streamlit) y **línea de comandos** (CLI). Ambos comparten el mismo backend.

```
┌──────────────────────────────────────────────────────┐
│  INTERFAZ WEB                    LÍNEA DE COMANDOS   │
│  streamlit run app.py            python -m src.main  │
│                                                      │
│  • Modo mock (sin dependencias)  • Control total     │
│  • Modo real (pipeline completo) • Logs detallados   │
│  • Chat interactivo              • Evaluación IR     │
│  • Ranking visual                • Scripts de bench  │
└──────────────────────────────────────────────────────┘
```

---

## 7. Interfaz web — Streamlit

### Arranque

```bash
# Desde la raíz del proyecto (con el entorno virtual activado)
streamlit run app.py
```

Se abre automáticamente en `http://localhost:8501`.

### Pantalla principal

```
┌─────────────────────────────────────────────────────────────────┐
│  BARRA LATERAL                │  ÁREA PRINCIPAL                 │
│                               │                                 │
│  🔍 RAG GitHub Skills         │  🔍 RAG GitHub Skills           │
│  TFG · Ingeniería del Software│  Sistema de recomendación...    │
│                               │                                 │
│  ── Añadir perfil ──          │  [Pantalla de bienvenida]       │
│  [ hwchase17      ] [Añadir]  │  o                              │
│                               │  [Historial de chat]           │
│  ── Perfiles (3) ──           │                                 │
│  ▶ @hwchase17                 │  > @hwchase17 — score 0.847     │
│  ▶ @jerryjliu                 │  > @jerryjliu — score 0.721     │
│  ▶ @pamelafox                 │                                 │
│                               │  Consulta> ____________ [Send]  │
│  ── Estado del sistema ──     │                                 │
│  Perfiles: 3  Evidencias: 456 │                                 │
│  ChromaDB: ● activo           │                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Paso a paso: añadir un perfil

1. Escribir el **username de GitHub** en el campo de la barra lateral (ej. `hwchase17`)
2. Pulsar **"➕ Añadir perfil"**
3. El sistema muestra el progreso de las 8 fases del pipeline en tiempo real
4. Al terminar, el perfil aparece en la lista con sus skills detectadas

### Paso a paso: hacer una consulta

1. Con al menos un perfil cargado, escribir la consulta en el chat inferior
2. El sistema muestra el progreso del pipeline RAG (6 pasos)
3. La respuesta incluye:
   - **Texto narrativo** con citas `[N]` a las evidencias
   - **Tabla de ranking** con scores por usuario
   - **Tarjetas de evidencias** con repo, fichero y fragmento de código

### Diferencia entre modo mock y modo real

| Característica | Mock (por defecto) | Real |
|---|---|---|
| GitHub Token | No necesario | Requerido |
| Ollama | No necesario | Requerido |
| Velocidad | ~5 segundos | 5-20 minutos por usuario |
| Datos | Generados aleatoriamente | Extraídos de GitHub |
| ChromaDB | No se usa | Se indexan los vectores |
| Para qué sirve | Demo, desarrollo UI | Producción, TFG |

Para activar el modo real, editar ambos ficheros de servicios:

```python
# services/profile_service.py  (línea ~35)
USE_MOCK: bool = False

# services/rag_service.py  (línea ~35)
USE_MOCK: bool = False
```

---

## 8. Pipeline por línea de comandos

El pipeline CLI es más potente que la interfaz web: permite controlar todos los parámetros, ver logs detallados y ejecutar la evaluación IR.

### Fase 1-4: Indexar un usuario

```bash
# Indexar el primer usuario de la lista (hwchase17 por defecto)
python -m src.main

# Indexar un usuario específico
python -m src.main hwchase17
python -m src.main jerryjliu
python -m src.main pamelafox

# Indexar todos los usuarios objetivo del TFG de una vez
python -m src.main --all
```

**Salida esperada:**
```
========================================================
PIPELINE — @hwchase17
========================================================
repos=6  files/repo=50  commits/repo=20  model=qwen2.5-coder:7b
--------------------------------------------------------
FASE 1 — INGESTA
--------------------------------------------------------
Conectado: hwchase17 (Harrison Chase)
Repositorios publicos: 47
Candidatos (sin forks ni archivados): 31
Seleccionados (top 6 por stars + actividad reciente):
  • hwchase17/langchain  [stars: 92000, último push: 2024-...]
  ...
Evidencias: 234  {'file': 178, 'commit': 32, 'issue': 14, 'pull_request': 10}
FASE 1 OK en 87.3s
--------------------------------------------------------
FASE 2 — PROCESAMIENTO (metricas + LLM)
--------------------------------------------------------
  [  1/234] file         src/langchain/chains/base.py  composite=0.84
    → 3 skills: ['LangChain', 'Pydantic', 'Python']
  [  2/234] file         src/langchain/vectorstores/chroma.py  composite=0.81
    → 2 skills: ['ChromaDB', 'LangChain']
  ...
FASE 2 OK en 312.4s
  Skills detectadas (18): ['ChromaDB', 'FastAPI', 'LangChain', ...]
--------------------------------------------------------
FASE 3 — PERFIL TECNICO
--------------------------------------------------------
FASE 3 OK en 0.2s
--------------------------------------------------------
FASE 4 — INDEXACION EN CHROMADB
--------------------------------------------------------
FASE 4 OK en 4.1s  (87 bloques)
  Total en ChromaDB: 87 bloques
    @hwchase17: 87 bloques, 18 skills
========================================================
COMPLETADO en 404.0s
  Evidencias: 234  |  Skills: 18  |  Bloques: 87
  Siguiente: python -m src.rag.query
========================================================
```

### Parámetros del pipeline (en `src/main.py`)

| Parámetro | Valor por defecto | Descripción |
|-----------|------------------|-------------|
| `max_repos` | 6 | Repositorios más relevantes a analizar |
| `max_files_per_repo` | 50 | Ficheros máximos por repositorio |
| `max_commits_per_repo` | 20 | Commits máximos por repositorio |
| `max_issues_per_repo` | 10 | Issues máximas por repositorio |
| `max_prs_per_repo` | 10 | Pull requests máximas por repositorio |
| `limit_files` | 30 | Ficheros `.py`/`.ipynb` enviados al LLM |

---

## 9. Sistema RAG — Consultas

### Consulta única

```bash
python -m src.rag.query "quien recomiendas para construir sistemas RAG?"
python -m src.rag.query "quien tiene experiencia con LangChain y LangGraph?"
python -m src.rag.query "busco a alguien que sepa integrar Azure OpenAI en Python"
```

**Salida esperada:**
```
------------------------------------------------------------
@jerryjliu es el candidato más adecuado para construir sistemas RAG [1][2].
Sus repositorios de LlamaIndex demuestran dominio profundo de indexación
vectorial y recuperación semántica [1]. Como segunda opción, @NirDiamant
cuenta con una colección extensa de tutoriales RAG con LangChain [3].

---
## Referencias
[1] @jerryjliu — **LlamaIndex** — `jerryjliu/llama_index / core/indices/vector_store.py`
     <https://github.com/jerryjliu/llama_index/blob/main/...>
[2] @jerryjliu — **RAG** — `jerryjliu/llama_index / docs/...`
     <https://github.com/jerryjliu/llama_index/blob/main/...>
[3] @NirDiamant — **LangChain** — `NirDiamant/RAG_Techniques / ...`
     <https://github.com/NirDiamant/RAG_Techniques/blob/main/...>
------------------------------------------------------------
```

### Modo interactivo

```bash
python -m src.rag.query
```

```
============================================================
  SISTEMA RAG - RECOMENDACION TECNICA DE DESARROLLADORES
============================================================
  ChromaDB: 432 bloques indexados
  Usuarios disponibles:
    @hwchase17   — 87 bloques, 18 skills unicas
    @jerryjliu   — 124 bloques, 22 skills unicas
    @pamelafox   — 76 bloques, 15 skills unicas
    @NirDiamant  — 89 bloques, 19 skills unicas
    @Shubhamsaboo— 56 bloques, 12 skills unicas

  Ejemplos de consultas:
    Quien recomiendas para construir sistemas RAG?
    Quien tiene mas experiencia con LangChain y LangGraph?
    Quien sabe implementar agentes LLM con herramientas?
  Escribe 'salir' para terminar
============================================================

Consulta> quien domina los embeddings semánticos?

[respuesta del sistema...]

Consulta> salir
[Sistema RAG detenido]
```

### Cómo funciona el pipeline RAG internamente

```
Consulta en lenguaje natural
        │
        ▼ embed_texts() — all-MiniLM-L6-v2
Embedding de la consulta [384 dims]
        │
        ▼ ChromaDB query — similitud coseno HNSW
30 bloques más relevantes (mezclados de todos los usuarios)
        │
        ▼ rank_candidates()
Agrupación por usuario:
  · Por skill: se queda con el bloque de mayor combined_score
  · combined_score = semantic_similarity × 0.40 + composite_score × 0.60
  · User score = media de skills + breadth_bonus (+0.02 por skill, máx +0.10)
        │
        ▼ generate_response()
Prompt RAG → Ollama (qwen2.5-coder:7b)
  · Contexto: hasta 3 candidatos × 4 skills = 12 evidencias numeradas
  · El LLM cita con [N] cada skill mencionada
  · Las referencias deterministas del pipeline reemplazan las del LLM
        │
        ▼
Respuesta con citas [N] + sección ## Referencias
```

### Degradación elegante sin Ollama

Si Ollama no está ejecutándose, el sistema sigue funcionando y devuelve una respuesta estructurada con los mismos datos:

```
Resultados para: «quien sabe de LangChain?»

No se pudo conectar con Ollama. Resultado estructurado:

#1 @hwchase17 (score 0.892) domina: LangChain, LangGraph, Python [1][2][3]
#2 @NirDiamant (score 0.741) domina: LangChain, RAG, ChromaDB [4][5][6]

----------------------------------------------------
Candidato                Skills    Score
----------------------------------------------------
@hwchase17               18        0.892  (LangChain, LangGraph, Python)
@NirDiamant              19        0.741  (LangChain, RAG, ChromaDB)
```

---

## 10. Evaluación del sistema

### Ejecutar la evaluación

```bash
# Evaluación estándar (ajusta a usuarios disponibles)
python -m src.evaluation.evaluator

# Modo estricto (solo evalúa consultas donde TODOS los usuarios relevantes están indexados)
python -m src.evaluation.evaluator --strict
```

### Métricas calculadas

| Métrica | Descripción | Valor objetivo |
|---------|-------------|----------------|
| **Precision@1** | Fracción de veces que el primer resultado es correcto | ≥ 0.70 |
| **Precision@3** | Fracción de correctos entre los 3 primeros resultados | ≥ 0.50 |
| **Recall@3** | Fracción de usuarios relevantes encontrados en top-3 | ≥ 0.60 |
| **MRR** | Reciprocal rank medio del primer resultado correcto | ≥ 0.70 |
| **NDCG@3** | Ganancia acumulada normalizada (pondera orden y relevancia) | ≥ 0.65 |

**Interpretación:**
- MRR ≥ 0.80 y P@1 ≥ 0.70 → **Excelente**
- MRR ≥ 0.60 y P@1 ≥ 0.50 → **Bueno** (sólido para un TFG)
- MRR ≥ 0.40 → **Aceptable**
- MRR < 0.40 → **Mejorable**

### Dataset de evaluación — Ground Truth

El sistema incluye 10 consultas de evaluación en `src/evaluation/ground_truth.py`:

| ID | Categoría | Consulta (resumida) | Usuarios relevantes |
|----|-----------|---------------------|---------------------|
| q001 | rag_systems | Sistemas RAG de recuperación de documentos | jerryjliu, NirDiamant, hwchase17 |
| q002 | rag_systems | Embeddings semánticos y búsqueda vectorial | jerryjliu, hwchase17, NirDiamant |
| q003 | orchestration | Experiencia con LangChain | hwchase17, NirDiamant |
| q004 | orchestration | LangGraph para flujos de agentes | hwchase17 |
| q005 | frameworks | Dominio de LlamaIndex | jerryjliu |
| q006 | llm_providers | Integración Azure OpenAI en Python | pamelafox |
| q007 | agents | Agentes LLM con herramientas personalizadas | hwchase17, Shubhamsaboo |
| q008 | deployment | Demos interactivas con Streamlit o Gradio | Shubhamsaboo, NirDiamant |
| q009 | deployment | Exponer un LLM como API REST con FastAPI | pamelafox, hwchase17 |
| q010 | education | Notebooks y tutoriales de IA generativa | NirDiamant, jerryjliu, Shubhamsaboo |

Cada consulta tiene relevancia graduada (0-3) para el cálculo de NDCG:
- **3** — Experto reconocido (autor del framework / proyecto principal)
- **2** — Conocedor avanzado (uso frecuente, contribuciones claras)
- **1** — Usuario básico (uso puntual o indirecto)

### Salida de la evaluación

```
============================================================
METRICAS GLOBALES (media sobre todas las consultas evaluadas)
============================================================
  Precision@1 : 0.8000
  Precision@3 : 0.5667
  Recall@3    : 0.7333
  MRR         : 0.8833
  NDCG@3      : 0.7481
  Latencia    : 0.234s por consulta
  Consultas   : 10
============================================================
  Interpretacion: EXCELENTE — el sistema es muy preciso

METRICAS POR CATEGORIA:
  Categoria              P@1    P@3    R@3    MRR   NDCG@3    N
  -----------------------------------------------------------------
  agents                1.000  0.667  0.833  1.000   0.847    2
  deployment            0.500  0.500  0.750  0.583   0.672    2
  education             1.000  0.667  0.667  1.000   0.889    1
  frameworks            1.000  0.667  0.556  1.000   0.792    1
  llm_providers         1.000  0.333  0.500  1.000   0.613    1
  orchestration         1.000  0.667  0.750  1.000   0.831    2
  rag_systems           0.500  0.417  0.722  0.750   0.624    2
```

El informe completo se guarda en `data/evaluation/report_YYYYMMDD_HHMMSS.json`.

---

## 11. Descripción de módulos

### `src/config.py`
Constantes globales del sistema (URL de Ollama, modelo LLM, modelo de embeddings, rutas de ChromaDB). Importar desde aquí evita valores duplicados y permite cambiar el modelo en un solo lugar.

---

### `src/main.py`
**Punto de entrada del pipeline CLI.** Coordina las 4 fases de indexación para un usuario. Cada fase está cronometrada y rodeada de manejo de errores para que un fallo no corrompa otros usuarios.

Funciones principales:
- `run_pipeline(username)` — Ejecuta las 4 fases completas
- `main()` — Parsea argumentos CLI (`--all`, username específico)

---

### `src/ingestion/`

#### `github_client.py`
Crea el cliente autenticado de PyGithub usando el token del `.env`. Lanza `ValueError` si el token no está configurado.

#### `schemas.py`
Define `GitHubEvidence`, el modelo Pydantic para todos los artefactos extraídos:

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `username` | str | Login del desarrollador analizado |
| `repo` | str | Nombre completo del repo (`owner/repo`) |
| `artifact_type` | str | `"file"` \| `"commit"` \| `"issue"` \| `"pull_request"` |
| `path` | str? | Ruta del fichero (solo para tipo `file`) |
| `author` | str? | Login del autor del artefacto |
| `text` | str | Contenido o mensaje (máx. 20 000 caracteres) |
| `metadata` | dict | URL, fechas, sha, estado, etc. |

#### `extract_repo.py`
Extrae artefactos de un repositorio individual:

- `walk_repository_files()` — Recorrido recursivo del árbol de directorios. Descarga solo ficheros con extensiones en `ALLOWED_EXTENSIONS` o nombres en `DEPENDENCY_FILES`/`DOC_FILES`. Excluye directorios en `SKIP_DIRS` (`.git`, `node_modules`, `venv`, etc.)
- `_extract_notebook_cells()` — Convierte notebooks `.ipynb` en solo el código y markdown que escribió el desarrollador (sin outputs ni metadatos JSON)
- `extract_commits()` — Commits del autor (filtrado por GitHub API, no local)
- `extract_issues()` — Issues creadas por el usuario (excluye PRs)
- `extract_pull_requests()` — PRs abiertas por el usuario (descarga hasta `max_prs × 3` para compensar que la API no filtra por creador)

**Ficheros siempre incluidos:** `requirements.txt`, `pyproject.toml`, `setup.py`, `environment.yml`, `package.json`, `Dockerfile`, `docker-compose.yml`, `.env.example`

**Extensiones de código soportadas:** `.py`, `.ipynb`, `.js`, `.ts`, `.tsx`, `.jsx`, `.md`, `.rst`, `.yaml`, `.yml`

#### `extract_user.py`
Orquesta la extracción de todos los repositorios de un usuario. Descarta forks y repos archivados, ordena por `(stars, actividad_reciente)` y selecciona los `max_repos` mejores.

---

### `src/processing/`

#### `metrics.py`
Calcula 4 métricas estructurales de calidad para cada evidencia, **sin LLM**:

| Métrica | Peso | Cálculo |
|---------|------|---------|
| `recency` | 15% | Decaimiento exponencial: `exp(−0.693 × días / 365)`. Hoy=1.0, 1 año=0.50, 2 años=0.25 |
| `authorship` | 20% | Commit=0.95, fichero propio=0.90, PR autor=0.85, issue autor=0.65 |
| `artifact_weight` | 30% | Python=0.85, TS=0.80, commit=0.80, PR=0.60, issue=0.45. Bonus tests (+0.10), penalización profundidad (−0.03/nivel) |
| `content_richness` | 35% | Ficheros: líneas + definiciones. Texto: longitud + bloques código |

`composite_score = Σ(peso_i × métrica_i)` → valor único en [0, 1].

#### `prompts.py`
Prompt LLM estricto para detección de skills y modelos Pydantic de validación:

- `SkillDetection` — una skill detectada: `skill` (1-80 chars), `explanation` (≤300 chars), `evidence_fragment` (≤1500 chars)
- `DetectionResponse` — `skills: list[SkillDetection]`
- `build_detection_prompt()` — Genera el prompt. Reglas principales: solo skills explícitas en el código, no basta con un import aislado, los manifiestos de dependencias sí cuentan como evidencia.

#### `code_filter.py`
Módulo central del procesamiento. Para cada evidencia cruda:
1. Calcula métricas estructurales con `score_evidence()`
2. Llama a Ollama con el prompt de detección
3. Parsea y valida la respuesta JSON del LLM (con reparación de JSON malformado)
4. Emite un bloque por skill detectada (todos con las mismas métricas)

Funciones de robustez:
- `_extract_raw_json()` — Extrae JSON de la respuesta aunque el LLM añada texto o fences Markdown
- `_parse_detection_response()` — Intenta parsear el JSON; si falla por saltos de línea literales dentro de strings, los repara y reintenta

---

### `src/profiling/`

#### `build_profile.py`
Agrega todos los bloques procesados de un usuario en un perfil técnico estructurado por skill. Para cada skill calcula:

- `evidence_count` — número de evidencias que la respaldan
- `avg_composite_score` — calidad media de las evidencias
- `evidence_diversity_score` — ratio de tipos de artefacto distintos (file, commit, issue, PR) que la respaldan
- `confidence` — `"alta"` si `avg_composite × min(count, 5) / 5 ≥ 0.65`, `"media"` si ≥ 0.35, `"baja"` en caso contrario

El perfil incluye hasta 8 ficheros de evidencia y 3 explicaciones de muestra por skill, ordenadas por score descendente.

---

### `src/vectorization/`

#### `embedder.py`
Singleton del modelo de embeddings `all-MiniLM-L6-v2`. La primera llamada descarga el modelo (~80 MB) y lo mantiene en memoria para las siguientes.

Función principal: `embed_texts(texts: List[str]) → List[List[float]]`

#### `indexer.py`
Gestiona la colección ChromaDB. El texto vectorizado para cada bloque combina:
```
Skill: <nombre_skill> | Evidence: <explicación_llm> | Code: <fragmento_código>
```

Los metadatos almacenados incluyen: username, skill, repo, path, artifact_type, url, las 4 métricas individuales, composite_score, explanation, evidence_fragment.

`index_user()` usa `upsert` para idempotencia: re-indexar no genera duplicados.

---

### `src/rag/`

#### `retrieval.py`
Convierte la consulta en embedding y recupera los bloques más similares de ChromaDB. La distancia coseno de ChromaDB está en [0, 2]. La conversión usa: `semantic_similarity = 1 − distancia / 2`.

#### `ranking.py`
Agrupa bloques por usuario y calcula el score final de cada candidato:

```
combined_score (por bloque) = semantic_similarity × 0.40 + composite_score × 0.60
score de skill              = max(combined_score) entre bloques de esa skill
avg_score (por usuario)     = media de los scores de todas las skills coincidentes
breadth_bonus               = min(n_skills × 0.02, 0.10)
final_score                 = min(avg_score + breadth_bonus, 1.0)
```

#### `generation.py`
Construye el prompt RAG con hasta 3 candidatos × 4 skills = hasta 12 evidencias numeradas. El LLM genera la narrativa citando con `[N]`. Las referencias deterministas del pipeline (repo, fichero, URL exactos) reemplazan las generadas por el LLM.

Si Ollama no está disponible: `_fallback_response()` devuelve la misma estructura sin LLM.

#### `query.py`
Punto de entrada del sistema RAG. Soporta consulta única (CLI) y modo interactivo. Coordina las tres fases (retrieval → ranking → generation) con logging y cronometrado.

---

### `src/evaluation/`

#### `ground_truth.py`
10 consultas con usuarios relevantes (relevancia binaria) y relevancia graduada (0-3) para NDCG. Cubre categorías: RAG, orquestación, frameworks, proveedores LLM, agentes, despliegue y educación.

#### `metrics.py`
Implementaciones puras (sin estado) de métricas IR estándar:
- `precision_at_k(retrieved, relevant, k)` — P@k
- `recall_at_k(retrieved, relevant, k)` — R@k
- `reciprocal_rank(retrieved, relevant)` — RR individual (para MRR)
- `ndcg_at_k(retrieved, grades, k)` — NDCG@k con relevancia graduada

#### `evaluator.py`
Ejecuta el pipeline completo (retrieval + ranking, sin generación LLM) sobre las 10 consultas. Calcula métricas por consulta, globales y por categoría. Guarda informe JSON en `data/evaluation/`.

---

### `src/utils/`

#### `logger.py`
Sistema de logging centralizado con dos canales:

- **Consola (INFO+):** mensajes concisos con colores ANSI (cian/verde/amarillo/rojo)
- **Fichero (DEBUG+):** traza completa con timestamp, módulo y nivel en `logs/pipeline_<usuario>_<timestamp>.log`

```python
# Uso en cualquier módulo:
from src.utils.logger import get_logger
log = get_logger(__name__)
log.info("Mensaje informativo")
log.debug("Detalle interno")
log.warning("Algo inesperado pero no fatal")
log.error("Fallo grave", exc_info=True)
```

`setup_pipeline_logger()` debe llamarse una vez desde `main.py` antes de iniciar el pipeline.

---

### `app.py`
Punto de entrada de la interfaz Streamlit. Gestiona el estado de sesión (`messages`, `profiles`, `last_ranking`, `last_evidences`), coordina la barra lateral (añadir perfiles) y el área de chat (historial + nueva consulta).

### `frontend/`

| Fichero | Responsabilidad |
|---------|-----------------|
| `layout.py` | Cabecera principal, header de barra lateral, métricas del sistema (perfiles, evidencias, estado ChromaDB) |
| `components.py` | Tarjetas de perfil, tabla de ranking, tarjetas de evidencias con fragmento de código, pantalla de bienvenida |
| `styles.py` | CSS personalizado inyectado en la app (tema oscuro, badges de tecnología, métricas animadas) |

### `services/`

| Fichero | Responsabilidad |
|---------|-----------------|
| `profile_service.py` | Pipeline de indexación de perfiles (mock y real). Generator que hace yield de pasos de progreso para la UI. En modo real llama a `src.main.run_pipeline()` |
| `rag_service.py` | Pipeline RAG de consultas (mock y real). Generator que hace yield de pasos. En modo real llama a `retrieve_blocks()`, `rank_candidates()` y `generate_response()` |

---

## 12. Formatos de datos

### JSONL crudo — `data/raw/<username>.jsonl`

Un objeto `GitHubEvidence` por línea:

```json
{
  "username": "hwchase17",
  "repo": "hwchase17/langchain",
  "artifact_type": "file",
  "path": "libs/langchain/langchain/chains/base.py",
  "author": null,
  "text": "from abc import ABC, abstractmethod\n...",
  "metadata": {
    "source": "github",
    "url": "https://github.com/hwchase17/langchain/blob/master/...",
    "size": 8432,
    "repo_pushed_at": "2024-11-15T10:23:41"
  }
}
```

### JSONL procesado — `data/procesado/<username>_processed.jsonl`

Un bloque por skill detectada (un mismo fichero puede generar N bloques si el LLM detecta N skills):

```json
{
  "username": "hwchase17",
  "skill": "LangChain",
  "source": {
    "repo": "hwchase17/langchain",
    "path": "libs/langchain/langchain/chains/base.py",
    "artifact_type": "file",
    "url": "https://github.com/hwchase17/langchain/blob/master/..."
  },
  "scores": {
    "recency": 0.9124,
    "authorship": 0.9,
    "artifact_weight": 0.82,
    "content_richness": 0.9,
    "composite_score": 0.8719
  },
  "explanation": "Clase base Chain de LangChain con método abstracto _call y soporte para callbacks.",
  "evidence_fragment": "class Chain(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):\n    ..."
}
```

### JSON de perfil — `data/perfiles/<username>_profile.json`

```json
{
  "user": "hwchase17",
  "profile_source": "data/procesado/hwchase17_processed.jsonl",
  "total_skill_blocks": 87,
  "unique_skills": 18,
  "technical_profile": {
    "hard_skills": [
      {
        "skill": "LangChain",
        "confidence": "alta",
        "evidence_count": 23,
        "avg_composite_score": 0.8341,
        "evidence_diversity_score": 0.75,
        "evidence_repositories": ["hwchase17/langchain", "hwchase17/langchain-experiments"],
        "evidence_files": ["libs/langchain/langchain/chains/base.py"],
        "artifact_types": ["commit", "file"],
        "sample_explanations": ["Clase base Chain con soporte para callbacks"],
        "sample_signals": ["class Chain(RunnableSerializable..."]
      }
    ]
  }
}
```

---

## 13. Benchmarking

El proyecto incluye scripts de evaluación comparativa de modelos:

```bash
# Comparar modelos LLM en la tarea de detección de skills
python benchmark_models.py

# Evaluación extendida de modelos
python eval_models.py

# Comparar modelos de embeddings para el retrieval
python eval_embeddings.py
```

Los resultados se guardan en:
- `benchmark_*.csv` y `benchmark_*.md` — resultados de modelos LLM
- `embedding_benchmark_*.csv` y `embedding_benchmark_*.md` — resultados de modelos de embeddings

Los benchmarks miden verdaderos positivos, falsos negativos, falsos positivos y trampas de comentarios (menciones sin uso real).

---

## 14. Ejemplos de consultas

```bash
# Por framework específico
python -m src.rag.query "quien tiene más experiencia con LangChain?"
python -m src.rag.query "quien domina LlamaIndex para indexación de documentos?"
python -m src.rag.query "quien sabe usar LangGraph para flujos de agentes?"

# Por proveedor de LLM
python -m src.rag.query "quien tiene experiencia integrando Azure OpenAI en Python?"
python -m src.rag.query "quien trabaja con modelos de Groq o Anthropic?"

# Por tipo de sistema
python -m src.rag.query "quien recomiendas para construir un sistema RAG completo?"
python -m src.rag.query "quien sabe de búsqueda semántica con embeddings?"
python -m src.rag.query "quien ha construido agentes LLM con herramientas personalizadas?"

# Por herramienta de despliegue
python -m src.rag.query "quien hace demos interactivas con Streamlit?"
python -m src.rag.query "quien sabe exponer un modelo LLM como API REST con FastAPI?"

# Por tipo de perfil
python -m src.rag.query "quien tiene más notebooks educativos sobre IA generativa?"
python -m src.rag.query "quien ha contribuido a frameworks open source de LLMs?"
```

---

## 15. Estructura de directorios

```
tfg-rag-github/
│
├── app.py                          # Punto de entrada de la interfaz Streamlit
├── requirements.txt                # Dependencias Python
├── .env                            # Token de GitHub (NO subir a git)
│
├── src/                            # Código fuente del backend
│   ├── config.py                   # Constantes globales (modelo, rutas)
│   ├── main.py                     # Pipeline CLI (fases 1-4)
│   │
│   ├── ingestion/                  # FASE 1: Extracción desde GitHub
│   │   ├── github_client.py        # Autenticación con la API
│   │   ├── schemas.py              # Modelo Pydantic GitHubEvidence
│   │   ├── extract_repo.py         # Ficheros, commits, issues, PRs
│   │   └── extract_user.py         # Selección y orquestación de repos
│   │
│   ├── processing/                 # FASE 2: Métricas + detección LLM
│   │   ├── metrics.py              # 4 métricas estructurales (sin LLM)
│   │   ├── prompts.py              # Prompt estricto + modelos Pydantic
│   │   └── code_filter.py          # Pipeline completo de procesamiento
│   │
│   ├── profiling/                  # FASE 3: Perfil técnico agregado
│   │   └── build_profile.py        # Agrega por skill, calcula confianza
│   │
│   ├── vectorization/              # FASE 4: Embeddings + ChromaDB
│   │   ├── embedder.py             # Singleton all-MiniLM-L6-v2
│   │   └── indexer.py              # Indexación y estadísticas de ChromaDB
│   │
│   ├── rag/                        # FASES 5-6: Retrieval + Ranking + Generación
│   │   ├── retrieval.py            # Búsqueda semántica en ChromaDB
│   │   ├── ranking.py              # Scoring multi-usuario con breadth bonus
│   │   ├── generation.py           # Prompt RAG → Ollama → respuesta con citas
│   │   └── query.py                # Punto de entrada del sistema RAG
│   │
│   ├── evaluation/                 # Evaluación IR del sistema
│   │   ├── ground_truth.py         # 10 consultas con relevancia graduada
│   │   ├── metrics.py              # P@k, Recall@k, MRR, NDCG@k (funciones puras)
│   │   └── evaluator.py            # Orquestación + informe JSON
│   │
│   └── utils/
│       └── logger.py               # Logging dual (consola coloreada + fichero)
│
├── frontend/                       # Componentes de la UI Streamlit
│   ├── layout.py                   # Cabecera y barra lateral
│   ├── components.py               # Tarjetas, ranking, evidencias
│   └── styles.py                   # CSS personalizado
│
├── services/                       # Lógica de negocio para la UI
│   ├── profile_service.py          # Pipeline de perfiles (mock/real)
│   └── rag_service.py              # Pipeline RAG (mock/real)
│
├── data/                           # Datos generados por el pipeline
│   ├── raw/                        # JSONL crudo (una línea = un artefacto)
│   ├── procesado/                  # JSONL procesado (una línea = un bloque/skill)
│   ├── perfiles/                   # JSON de perfil técnico por usuario
│   ├── chromadb/                   # Índice vectorial persistente (HNSW)
│   └── evaluation/                 # Informes de evaluación JSON
│
├── logs/                           # Logs del pipeline con timestamp
│
├── benchmark_models.py             # Benchmarking de modelos LLM
├── eval_models.py                  # Evaluación extendida de modelos
└── eval_embeddings.py              # Benchmarking de modelos de embeddings
```

---

## 16. Solución de problemas

### El pipeline falla con "No se encontró GITHUB_TOKEN"

Verificar que el `.env` existe y tiene el token:
```bash
cat .env
# Debe mostrar: GITHUB_TOKEN=ghp_xxxxx
```

### Ollama no responde / timeout

```bash
# Verificar que Ollama está ejecutándose
curl http://localhost:11434/api/tags

# Arrancar Ollama si no está activo
ollama serve

# Verificar que el modelo está descargado
ollama list
# Si no aparece qwen2.5-coder:7b:
ollama pull qwen2.5-coder:7b
```

> El sistema funciona sin Ollama: devuelve una respuesta estructurada en lugar de narrativa.

### "ChromaDB está vacío"

Es necesario indexar al menos un usuario antes de hacer consultas:

```bash
python -m src.main hwchase17
# Esperar a que completen las 4 fases (~5-15 minutos)
```

### La Fase 2 tarda mucho

La Fase 2 llama al LLM una vez por evidencia. Con 234 evidencias y `qwen2.5-coder:7b`, puede tardar 5-15 minutos dependiendo del hardware. Para pruebas rápidas, reducir `limit_files` en `src/main.py`:

```python
limit_files = 10   # en lugar de 30
```

### La interfaz web no muestra datos reales

Por defecto, los servicios usan `USE_MOCK = True`. Para datos reales:

```python
# services/profile_service.py  (línea ~35)
USE_MOCK: bool = False

# services/rag_service.py  (línea ~35)
USE_MOCK: bool = False
```

### Re-indexar un usuario desde cero

ChromaDB usa `upsert`, así que re-ejecutar el pipeline actualiza los datos sin duplicar. Para borrar completamente un usuario del índice:

```python
import chromadb
client = chromadb.PersistentClient(path="data/chromadb")
col = client.get_collection("technical_blocks")
result = col.get(where={"username": {"$eq": "hwchase17"}})
if result["ids"]:
    col.delete(ids=result["ids"])
```

### Los logs del modo `--all` se mezclan

Es un comportamiento conocido: en modo `--all`, todos los usuarios comparten el fichero de log del primero. Para tener logs separados, indexar cada usuario en comandos separados:

```bash
python -m src.main hwchase17
python -m src.main jerryjliu
python -m src.main pamelafox
```

---

## Usuarios objetivo del TFG

| Usuario | Perfil técnico | Skills principales |
|---------|----------------|-------------------|
| [@hwchase17](https://github.com/hwchase17) | Creador de LangChain y LangGraph | LangChain, LangGraph, agentes LLM, LCEL, Python |
| [@jerryjliu](https://github.com/jerryjliu) | Creador de LlamaIndex | LlamaIndex, RAG, embeddings, índices vectoriales |
| [@pamelafox](https://github.com/pamelafox) | Microsoft Developer Advocate | Azure OpenAI, FastAPI, Python, demos IA |
| [@NirDiamant](https://github.com/NirDiamant) | Tutoriales RAG y agentes | RAG, LangChain, ChromaDB, notebooks |
| [@Shubhamsaboo](https://github.com/Shubhamsaboo) | Apps de agentes IA | Streamlit, Groq, agentes LLM, interfaces IA |

---

*TFG · Ingeniería del Software — Sistema de recomendación técnica basado en evidencias de GitHub*

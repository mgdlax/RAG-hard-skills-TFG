"""
Benchmark de modelos de embeddings para el sistema RAG de RRHH.

Compara:
  · intfloat/multilingual-e5-small                      (multilingüe, 384 dims, prefijo query:/passage:)
  · isuruwijesiri/all-MiniLM-L6-v2-code-search-512      (code search, 384 dims, sin prefijo)

Corpus:
  Construido desde data/procesado/HalemoGPA_processed.jsonl + 2 documentos sintéticos.
  Formato documental: "Skill: {skill} | Evidence: {explanation} | Code: {fragment}"

Casos de prueba (16)
────────────────────
  C01  rag_real_format    · IA médica con PyTorch
  C02  rag_real_format    · Interfaz web con Streamlit
  C03  rag_real_format    · Explicabilidad visual con Grad-CAM
  C04  rag_real_format    · Procesamiento de imágenes con OpenCV
  C05  false_positive_trap· Despliegue con Docker
  C06  rag_real_format    · Backend API con FastAPI
  C07  profile_query      · Frontend con React y TypeScript
  C08  rag_real_format    · Testing de componentes React
  C09  rag_real_format    · Base de datos SQL (PostgreSQL)
  C10  rag_real_format    · Generación de informes PDF (reportlab)
  C11  rag_real_format    · Peticiones HTTP desde frontend (Axios)
  C12  rag_real_format    · Gestión de estado frontend (Redux)
  C13  synthetic_rag      · Sistema RAG con LangChain
  C14  synthetic_agents   · Agentes IA con LangGraph
  C15  code_only          · Código puro: PyTorch training
  C16  cross_lingual      · Consulta ES: JavaScript y DOM

Métricas por caso
─────────────────
  hit@1        1 si el primer resultado del ranking es relevante
  mrr          reciprocal rank del primer documento relevante
  precision@3  proporción de relevantes en el top 3
  recall@3     fracción de relevantes recuperados en el top 3
  max_sim_rel  similitud máxima entre query y docs relevantes
  max_sim_irr  similitud máxima entre query y docs irrelevantes
  gap          max_sim_rel − max_sim_irr  (>0 = modelo discrimina bien)

Score global
────────────
  0.35 × hit_rate + 0.25 × MRR + 0.20 × P@3 + 0.10 × R@3 + 0.10 × norm(gap)
  donde norm(gap) = min(max(avg_gap × 5, 0), 1)

Salida
──────
  · Consola con detalle por caso y tabla resumen
  · embedding_benchmark_results_<timestamp>.csv
  · embedding_benchmark_summary_<timestamp>.md

Uso
───
  pip install sentence-transformers numpy
  python eval_embeddings.py
"""

import csv
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
#  Configuración
# ─────────────────────────────────────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "multilingual-e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    "minilm-code-search-512": {
        "name": "isuruwijesiri/all-MiniLM-L6-v2-code-search-512",
        "query_prefix": "",
        "doc_prefix": "",
    },
}

PROCESSED_PATH = Path("data/procesado/HalemoGPA_processed.jsonl")
OUTPUT_DIR = Path(".")
RANDOM_SEED = 42
MAX_IRRELEVANT = 8

_EXPLANATION_MAX_CHARS = 200
_FRAGMENT_MAX_CHARS = 300

# ─────────────────────────────────────────────────────────────────────────────
#  Corpus
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_document_text(record: dict) -> str:
    skill = record.get("skill", "")
    explanation = record.get("explanation", "")[:_EXPLANATION_MAX_CHARS]
    fragment = record.get("evidence_fragment", "")[:_FRAGMENT_MAX_CHARS]

    parts = [f"Skill: {skill}"]
    if explanation:
        parts.append(f"Evidence: {explanation}")
    if fragment:
        parts.append(f"Code: {fragment}")

    return " | ".join(parts)


SYNTHETIC_RECORDS: list[dict] = [
    {
        "skill": "LangChain",
        "explanation": "The project builds a RAG pipeline using retrievers, embeddings and a vector database.",
        "evidence_fragment": (
            "retriever = vectorstore.as_retriever()\n"
            "qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)"
        ),
        "username": "synthetic",
        "source": {
            "repo": "synthetic/rag-demo",
            "path": "src/rag.py",
            "artifact_type": "file",
            "url": "",
        },
        "scores": {"composite_score": 1.0},
    },
    {
        "skill": "LangGraph",
        "explanation": (
            "The project defines a multi-agent workflow using graph nodes, "
            "conditional edges and shared state between agents."
        ),
        "evidence_fragment": (
            "workflow.add_node('research_agent', research_agent)\n"
            "workflow.add_conditional_edges('supervisor', route_next_agent)"
        ),
        "username": "synthetic",
        "source": {
            "repo": "synthetic/agent-demo",
            "path": "src/graph.py",
            "artifact_type": "file",
            "url": "",
        },
        "scores": {"composite_score": 1.0},
    },
]


def build_corpus(records: list[dict]) -> list[dict]:
    corpus = []
    for i, record in enumerate(records):
        skill = record.get("skill", "").strip()
        if not skill:
            continue
        explanation = record.get("explanation", "").strip()
        fragment = record.get("evidence_fragment", "").strip()
        if not explanation and not fragment:
            print(f"  [WARN] Registro {i} (skill={skill!r}) sin texto útil. Omitido.")
            continue
        corpus.append({
            "doc_id": f"{record.get('username', 'unk')}_{i:04d}",
            "text": build_document_text(record),
            "skill": skill,
            "username": record.get("username", ""),
            "repo": record.get("source", {}).get("repo", ""),
            "artifact_type": record.get("source", {}).get("artifact_type", ""),
            "url": record.get("source", {}).get("url", ""),
            "composite_score": record.get("scores", {}).get("composite_score", 0.0),
        })
    return corpus


# ─────────────────────────────────────────────────────────────────────────────
#  Casos de prueba
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingCase:
    id: str
    name: str
    tipo: str
    query: str
    relevant_skills: list[str]
    irrelevant_skills: list[str] = field(default_factory=list)
    note: str = ""


CASES: list[EmbeddingCase] = [
    EmbeddingCase(
        id="C01",
        name="IA médica con PyTorch",
        tipo="rag_real_format",
        query="Busco un perfil con experiencia en modelos de IA para clasificación de imágenes médicas MRI",
        relevant_skills=["PyTorch"],
        irrelevant_skills=["JavaScript", "CSS", "React"],
        note="Debe recuperar evidencias de PyTorch relacionadas con clasificación de tumores MRI.",
    ),
    EmbeddingCase(
        id="C02",
        name="Interfaz web para aplicación de IA",
        tipo="rag_real_format",
        query="Necesito alguien que pueda crear una interfaz web interactiva para mostrar resultados de un modelo de IA",
        relevant_skills=["Streamlit"],
        irrelevant_skills=["reportlab", "urllib3", "requests"],
        note="Debe recuperar Streamlit frente a librerías auxiliares.",
    ),
    EmbeddingCase(
        id="C03",
        name="Explicabilidad visual en visión por computador",
        tipo="rag_real_format",
        query="Busco experiencia en explicabilidad visual para redes neuronales de imágenes usando mapas de calor",
        relevant_skills=["Grad-CAM"],
        irrelevant_skills=["Docker", "requests", "CSS"],
        note="Debe recuperar Grad-CAM.",
    ),
    EmbeddingCase(
        id="C04",
        name="Procesamiento de imágenes con OpenCV",
        tipo="rag_real_format",
        query="Necesito experiencia procesando imágenes con OpenCV en Python",
        relevant_skills=["OpenCV", "OpenCV-Python-Headless"],
        irrelevant_skills=["PyTorch", "Streamlit", "reportlab"],
        note="Debe distinguir OpenCV de PyTorch.",
    ),
    EmbeddingCase(
        id="C05",
        name="Despliegue con Docker",
        tipo="false_positive_trap",
        query="Busco un desarrollador con experiencia creando contenedores Docker y Dockerfile para desplegar aplicaciones",
        relevant_skills=["Docker"],
        irrelevant_skills=["HF Spaces", "Streamlit", "uvicorn"],
        note="Debe priorizar Docker frente a despliegue genérico o Streamlit Cloud.",
    ),
    EmbeddingCase(
        id="C06",
        name="Backend API con FastAPI",
        tipo="rag_real_format",
        query="Necesito desarrollar una API backend en Python con FastAPI",
        relevant_skills=["FastAPI"],
        irrelevant_skills=["React", "CSS", "PostgreSQL"],
        note="Debe recuperar FastAPI.",
    ),
    EmbeddingCase(
        id="C07",
        name="Frontend con React y TypeScript",
        tipo="profile_query",
        query="Busco un perfil frontend con experiencia en React, componentes y tipado con TypeScript",
        relevant_skills=["React", "TypeScript"],
        irrelevant_skills=["PostgreSQL", "Docker", "PyTorch"],
        note="Debe recuperar evidencias frontend.",
    ),
    EmbeddingCase(
        id="C08",
        name="Testing de componentes React",
        tipo="rag_real_format",
        query="Necesito experiencia escribiendo tests unitarios de componentes React con Jest y React Testing Library",
        relevant_skills=["Jest", "React Testing Library"],
        irrelevant_skills=["CSS", "JavaScript", "Redux"],
        note="Debe recuperar testing real, no solo frontend genérico.",
    ),
    EmbeddingCase(
        id="C09",
        name="Base de datos SQL en Python",
        tipo="rag_real_format",
        query="Busco experiencia conectando aplicaciones Python con bases de datos SQL PostgreSQL",
        relevant_skills=["PostgreSQL"],
        irrelevant_skills=["Streamlit", "CSS", "React"],
        note="Debe recuperar PostgreSQL y psycopg2.",
    ),
    EmbeddingCase(
        id="C10",
        name="Generación de informes PDF",
        tipo="rag_real_format",
        query="Necesito generar informes PDF personalizados desde Python",
        relevant_skills=["reportlab"],
        irrelevant_skills=["PyTorch", "Streamlit", "Docker"],
        note="Debe recuperar reportlab.",
    ),
    EmbeddingCase(
        id="C11",
        name="Peticiones HTTP desde frontend",
        tipo="rag_real_format",
        query="Busco experiencia haciendo peticiones HTTP desde una aplicación frontend",
        relevant_skills=["Axios"],
        irrelevant_skills=["requests", "urllib3", "PostgreSQL"],
        note="Debe distinguir Axios (frontend) de requests/urllib3 (backend).",
    ),
    EmbeddingCase(
        id="C12",
        name="Gestión de estado frontend",
        tipo="rag_real_format",
        query="Necesito experiencia gestionando estado global en aplicaciones frontend",
        relevant_skills=["Redux"],
        irrelevant_skills=["React Router", "Axios", "CSS"],
        note="Debe recuperar Redux frente a otras librerías frontend.",
    ),
    EmbeddingCase(
        id="C13",
        name="Sistema RAG con base vectorial y LLM",
        tipo="synthetic_rag",
        query="Busco un desarrollador para crear sistemas RAG con embeddings, bases vectoriales y LLMs",
        relevant_skills=["LangChain"],
        irrelevant_skills=["FastAPI", "Streamlit", "PyTorch"],
        note="Caso central del TFG. Documento sintético.",
    ),
    EmbeddingCase(
        id="C14",
        name="Agentes de IA y workflows multiagente",
        tipo="synthetic_agents",
        query="Busco un perfil para desarrollar agentes de IA con herramientas y flujos multiagente",
        relevant_skills=["LangGraph"],
        irrelevant_skills=["Streamlit", "React", "FastAPI"],
        note="Comprueba si el modelo relaciona agentes IA con LangGraph. Documento sintético.",
    ),
    EmbeddingCase(
        id="C15",
        name="Código puro: PyTorch training",
        tipo="code_only",
        query="deep learning model training with PyTorch",
        relevant_skills=["PyTorch"],
        irrelevant_skills=["JavaScript", "CSS", "PostgreSQL"],
        note="Observar si el modelo de code search mejora cuando la query es en inglés y técnica.",
    ),
    EmbeddingCase(
        id="C16",
        name="Cross-lingual: JavaScript y DOM",
        tipo="cross_lingual",
        query="Busco experiencia desarrollando aplicaciones web interactivas con JavaScript y manipulación del DOM",
        relevant_skills=["JavaScript"],
        irrelevant_skills=["PyTorch", "Docker", "PostgreSQL"],
        note="Consulta en español frente a evidencias en inglés/español. Prueba ventaja de multilingual-e5.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
#  Estructuras de datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedDoc:
    doc_id: str
    skill: str
    is_relevant: bool
    sim: float
    repo: str
    artifact_type: str
    url: str


@dataclass
class CaseResult:
    model_key: str
    case: EmbeddingCase
    ranking: list  # list[RankedDoc], ordenado por sim desc
    hit_at_1: bool
    mrr: float
    precision_at_3: float
    recall_at_3: float
    max_sim_relevant: float
    max_sim_irrelevant: float
    gap: float
    encode_ms: float


@dataclass
class ModelSummary:
    model_key: str
    hit_rate: float
    avg_gap: float
    avg_mrr: float
    avg_precision_at_3: float
    avg_recall_at_3: float
    avg_sim_relevant: float
    avg_sim_irrelevant: float
    avg_encode_ms: float
    results: list = field(default_factory=list)  # list[CaseResult]

    @property
    def score(self) -> float:
        normalized_avg_gap = min(max(self.avg_gap * 5, 0.0), 1.0)
        return round(
            0.35 * self.hit_rate
            + 0.25 * self.avg_mrr
            + 0.20 * self.avg_precision_at_3
            + 0.10 * self.avg_recall_at_3
            + 0.10 * normalized_avg_gap,
            4,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Selección de candidatos (precomputada una vez, compartida por todos los modelos)
# ─────────────────────────────────────────────────────────────────────────────

def select_candidates(
    case: EmbeddingCase,
    corpus: list[dict],
    rng: random.Random,
    max_irrelevant: int = MAX_IRRELEVANT,
) -> dict:
    """
    Devuelve {"docs": list[dict], "relevant_ids": set[str]}.
    Todos los documentos relevantes se incluyen siempre.
    Los irrelevantes se seleccionan priorizando las skills en irrelevant_skills.
    """
    relevant_docs = [d for d in corpus if d["skill"] in case.relevant_skills]
    relevant_ids = {d["doc_id"] for d in relevant_docs}

    if not relevant_docs:
        return {"docs": [], "relevant_ids": set()}

    preferred_irr = [
        d for d in corpus
        if d["skill"] in case.irrelevant_skills and d["doc_id"] not in relevant_ids
    ]
    other_irr = [
        d for d in corpus
        if d["skill"] not in case.relevant_skills
        and d["skill"] not in case.irrelevant_skills
        and d["doc_id"] not in relevant_ids
    ]

    rng.shuffle(preferred_irr)
    rng.shuffle(other_irr)
    irrelevant_docs = (preferred_irr + other_irr)[:max_irrelevant]

    all_docs = relevant_docs + irrelevant_docs
    return {"docs": all_docs, "relevant_ids": relevant_ids}


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluación
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    config: dict,
    corpus: list[dict],
    case_candidates: dict,
) -> list[CaseResult]:
    print(f"\n  Cargando {config['name']}...", flush=True)
    model = SentenceTransformer(config["name"])
    qp = config["query_prefix"]
    dp = config["doc_prefix"]

    results = []
    for case in CASES:
        if case.id not in case_candidates:
            continue
        info = case_candidates[case.id]
        candidate_docs: list[dict] = info["docs"]
        relevant_ids: set = info["relevant_ids"]

        if not relevant_ids:
            print(f"  [WARN] {case.id}: sin documentos relevantes. Caso omitido.", flush=True)
            continue
        if not candidate_docs:
            print(f"  [WARN] {case.id}: sin candidatos. Caso omitido.", flush=True)
            continue

        query_text = qp + case.query
        doc_texts = [dp + d["text"] for d in candidate_docs]
        all_texts = [query_text] + doc_texts

        t0 = time.monotonic()
        embeddings = model.encode(
            all_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000 / len(all_texts)

        q_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Con normalize_embeddings=True el producto escalar es la similitud coseno
        sims = [float(np.dot(q_emb, doc_embs[i])) for i in range(len(candidate_docs))]

        ranked_pairs = sorted(zip(candidate_docs, sims), key=lambda x: x[1], reverse=True)
        ranking = [
            RankedDoc(
                doc_id=doc["doc_id"],
                skill=doc["skill"],
                is_relevant=doc["doc_id"] in relevant_ids,
                sim=sim,
                repo=doc["repo"],
                artifact_type=doc["artifact_type"],
                url=doc.get("url", ""),
            )
            for doc, sim in ranked_pairs
        ]

        total_relevant = len(relevant_ids)

        hit_at_1 = ranking[0].is_relevant if ranking else False

        mrr = 0.0
        for i, rd in enumerate(ranking):
            if rd.is_relevant:
                mrr = 1.0 / (i + 1)
                break

        top3 = ranking[:3]
        relevant_in_top3 = sum(1 for rd in top3 if rd.is_relevant)
        precision_at_3 = relevant_in_top3 / min(3, len(top3)) if top3 else 0.0
        recall_at_3 = min(relevant_in_top3 / total_relevant, 1.0) if total_relevant > 0 else 0.0

        rel_sims = [rd.sim for rd in ranking if rd.is_relevant]
        irr_sims = [rd.sim for rd in ranking if not rd.is_relevant]
        max_sim_rel = max(rel_sims) if rel_sims else 0.0
        max_sim_irr = max(irr_sims) if irr_sims else 0.0
        gap = max_sim_rel - max_sim_irr

        results.append(CaseResult(
            model_key=model_key,
            case=case,
            ranking=ranking,
            hit_at_1=hit_at_1,
            mrr=mrr,
            precision_at_3=precision_at_3,
            recall_at_3=recall_at_3,
            max_sim_relevant=max_sim_rel,
            max_sim_irrelevant=max_sim_irr,
            gap=gap,
            encode_ms=elapsed_ms,
        ))

    return results


def compute_summary(model_key: str, results: list[CaseResult]) -> ModelSummary:
    n = len(results)
    if n == 0:
        return ModelSummary(
            model_key=model_key,
            hit_rate=0.0, avg_gap=0.0, avg_mrr=0.0,
            avg_precision_at_3=0.0, avg_recall_at_3=0.0,
            avg_sim_relevant=0.0, avg_sim_irrelevant=0.0,
            avg_encode_ms=0.0,
        )
    return ModelSummary(
        model_key=model_key,
        hit_rate=sum(1 for r in results if r.hit_at_1) / n,
        avg_gap=sum(r.gap for r in results) / n,
        avg_mrr=sum(r.mrr for r in results) / n,
        avg_precision_at_3=sum(r.precision_at_3 for r in results) / n,
        avg_recall_at_3=sum(r.recall_at_3 for r in results) / n,
        avg_sim_relevant=sum(r.max_sim_relevant for r in results) / n,
        avg_sim_irrelevant=sum(r.max_sim_irrelevant for r in results) / n,
        avg_encode_ms=sum(r.encode_ms for r in results) / n,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Presentación en consola
# ─────────────────────────────────────────────────────────────────────────────

W = 112


def sep(c: str = "─") -> None:
    print(c * W)


def bar(v: float, width: int = 10) -> str:
    v_norm = max(0.0, min(1.0, v))
    filled = round(v_norm * width)
    return "█" * filled + "░" * (width - filled)


def print_case_results(
    case: EmbeddingCase,
    results_by_model: dict,
) -> None:
    sep("═")
    print(f"  {case.id} · [{case.tipo}] · {case.name}")
    print(f"  Query  : {case.query[:105]}")
    if case.note:
        print(f"  Nota   : {case.note}")
    sep("·")

    for mk, r in results_by_model.items():
        hit_icon = "✓" if r.hit_at_1 else "✗"
        print(
            f"\n  {mk:<32}  Hit@1={hit_icon}  MRR={r.mrr:.3f}  "
            f"P@3={r.precision_at_3:.2f}  R@3={r.recall_at_3:.2f}  "
            f"Gap={r.gap:+.4f}  enc={r.encode_ms:.1f}ms"
        )
        print(
            f"  {'':32}  MaxSimRel={r.max_sim_relevant:.4f}  "
            f"MaxSimIrr={r.max_sim_irrelevant:.4f}"
        )
        for rank_i, rd in enumerate(r.ranking[:3], start=1):
            tag = "[REL]" if rd.is_relevant else "[IRR]"
            print(
                f"    #{rank_i} {tag:<6} skill={rd.skill:<28} "
                f"sim={rd.sim:.4f} {bar(rd.sim)}  "
                f"repo={rd.repo[:38]}  type={rd.artifact_type}"
            )
    print()


def print_summary(summaries: list[ModelSummary]) -> None:
    sep("═")
    print("  RESUMEN GLOBAL")
    sep("═")
    print(
        f"  {'Modelo':<32}  {'Hit@1':>6}  {'MRR':>6}  "
        f"{'P@3':>5}  {'R@3':>5}  {'AvgGap':>7}  {'ms/enc':>7}  {'Score':>7}"
    )
    sep("·")
    winner = max(summaries, key=lambda s: s.score)
    for s in sorted(summaries, key=lambda x: x.score, reverse=True):
        medal = "★ " if s.model_key == winner.model_key else "  "
        print(
            f"  {medal}{s.model_key:<30}  "
            f"{s.hit_rate * 100:>5.1f}%  "
            f"{s.avg_mrr:>6.3f}  "
            f"{s.avg_precision_at_3:>5.3f}  "
            f"{s.avg_recall_at_3:>5.3f}  "
            f"{s.avg_gap:>+7.4f}  "
            f"{s.avg_encode_ms:>6.1f}ms  "
            f"{s.score:>7.4f}"
        )
    sep("═")
    n_hit = sum(1 for r in winner.results if r.hit_at_1)
    print(f"\n  Modelo con mejor score : {winner.model_key}  (score={winner.score:.4f})")
    print(f"  Hit@1 rate             : {winner.hit_rate * 100:.1f}%  ({n_hit}/{len(winner.results)} casos)")
    print(f"  MRR medio              : {winner.avg_mrr:.3f}")
    print(f"  Precision@3 medio      : {winner.avg_precision_at_3:.3f}")
    print(f"  Gap medio              : {winner.avg_gap:+.4f}")
    print(f"  Latencia media         : {winner.avg_encode_ms:.1f}ms por texto")
    sep("═")


# ─────────────────────────────────────────────────────────────────────────────
#  Exportación
# ─────────────────────────────────────────────────────────────────────────────

def export_csv(all_results: list[CaseResult], path: Path) -> None:
    rows = []
    for r in all_results:
        top3 = r.ranking[:3]
        top1 = top3[0] if top3 else None
        rows.append({
            "modelo":             r.model_key,
            "caso_id":            r.case.id,
            "caso_nombre":        r.case.name,
            "tipo":               r.case.tipo,
            "query":              r.case.query,
            "hit_1":              int(r.hit_at_1),
            "mrr":                round(r.mrr, 6),
            "precision_at_3":     round(r.precision_at_3, 6),
            "recall_at_3":        round(r.recall_at_3, 6),
            "max_sim_relevant":   round(r.max_sim_relevant, 6),
            "max_sim_irrelevant": round(r.max_sim_irrelevant, 6),
            "gap":                round(r.gap, 6),
            "encode_ms":          round(r.encode_ms, 2),
            "top1_skill":         top1.skill if top1 else "",
            "top1_is_relevant":   int(top1.is_relevant) if top1 else "",
            "top1_score":         round(top1.sim, 6) if top1 else "",
            "top1_repo":          top1.repo if top1 else "",
            "top3_skills":        ";".join(rd.skill for rd in top3),
        })

    if not rows:
        print("  [WARN] No hay resultados para exportar en CSV.")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  → CSV guardado en: {path}")


def _auto_conclusion(winner: ModelSummary, summaries: list[ModelSummary]) -> list[str]:
    others = [s for s in summaries if s.model_key != winner.model_key]
    score_gap = winner.score - min(s.score for s in others) if others else 0.0
    if score_gap > 0.05:
        gap_note = "La ventaja es significativa respecto al resto de modelos."
    else:
        gap_note = "La diferencia es pequeña — todos los modelos son competitivos en este corpus."

    lines: list[str] = [
        f"El modelo con mejor score es **`{winner.model_key}`** "
        f"(score={winner.score:.4f}, Δ={score_gap:+.4f} respecto al siguiente). {gap_note}",
        "",
    ]

    wk = winner.model_key
    if "multilingual" in wk:
        lines.append(
            "El rendimiento de `multilingual-e5-small` puede deberse a que las consultas están en "
            "español y las evidencias mezclan español, inglés y código. Este modelo está entrenado "
            "en múltiples idiomas y sus prefijos `query:` / `passage:` optimizan la asimetría "
            "consulta-documento propia del retrieval semántico."
        )
    elif "code-search" in wk:
        lines.append(
            "El rendimiento de `minilm-code-search-512` puede deberse a que está afinado para "
            "búsqueda semántica de código, lo que le permite aprovechar mejor el campo `Code` "
            "presente en los documentos. Aunque es un modelo ligero, su especialización puede "
            "compensar la falta de soporte multilingüe en consultas en español."
        )
    else:
        lines.append(
            "El modelo ganador indica que el formato enriquecido "
            "`Skill: … | Evidence: … | Code: …` aporta suficiente contexto semántico "
            "para que el retrieval sea efectivo sobre este corpus."
        )

    lines += [
        "",
        "**Advertencia de alcance:** estos resultados dependen del corpus reducido "
        f"({sum(1 for _ in CASES)} casos de prueba, corpus de menos de 50 evidencias reales más "
        "2 documentos sintéticos), las consultas definidas manualmente y el formato documental "
        "del prototipo. No deben interpretarse como una evaluación universal de los modelos. "
        "Su propósito es justificar la elección del modelo de embeddings para ChromaDB "
        "en el contexto concreto del sistema RAG del TFG.",
    ]
    return lines


def export_markdown(
    summaries: list[ModelSummary],
    path: Path,
    timestamp: str,
) -> None:
    winner = max(summaries, key=lambda s: s.score)

    case_by_id: dict = {}
    for s in summaries:
        for r in s.results:
            case_by_id[r.case.id] = r.case
    evaluated_cases = sorted(case_by_id.values(), key=lambda c: c.id)

    lines: list[str] = [
        "# Benchmark · Modelos de Embeddings para RAG de RRHH",
        "",
        f"**Fecha:** {timestamp}  ",
        f"**Modelos evaluados:** {' · '.join(s.model_key for s in summaries)}  ",
        f"**Casos evaluados:** {len(evaluated_cases)}  ",
        f"**Corpus:** `{PROCESSED_PATH}` + 2 documentos sintéticos (LangChain, LangGraph)  ",
        "**Formato documental:** `Skill: … | Evidence: … | Code: …`  ",
        "",
        "---",
        "",
        "## Resumen global",
        "",
        "| Modelo | Hit@1 | MRR | P@3 | R@3 | Avg Gap | ms/enc | Score |",
        "|--------|-------|-----|-----|-----|---------|--------|-------|",
    ]

    for s in sorted(summaries, key=lambda x: x.score, reverse=True):
        medal = "🥇 " if s.model_key == winner.model_key else ""
        lines.append(
            f"| {medal}`{s.model_key}` "
            f"| {s.hit_rate * 100:.1f}% "
            f"| {s.avg_mrr:.3f} "
            f"| {s.avg_precision_at_3:.3f} "
            f"| {s.avg_recall_at_3:.3f} "
            f"| {s.avg_gap:+.4f} "
            f"| {s.avg_encode_ms:.1f}ms "
            f"| **{s.score:.4f}** |"
        )

    lines += [
        "",
        "> **Score** = 0.35 × Hit@1 + 0.25 × MRR + 0.20 × P@3 + 0.10 × R@3 + 0.10 × norm(AvgGap)  ",
        "> norm(AvgGap) = min(max(AvgGap × 5, 0), 1)",
        "",
        f"**Modelo ganador:** `{winner.model_key}` con score {winner.score:.4f}",
        "",
        "---",
        "",
        "## Análisis por tipo de caso",
        "",
        "| Tipo | Modelo | Hit@1 | MRR | P@3 | Avg Gap | N casos |",
        "|------|--------|-------|-----|-----|---------|---------|",
    ]

    tipos = sorted({c.tipo for c in evaluated_cases})
    for tipo in tipos:
        for s in summaries:
            rs = [r for r in s.results if r.case.tipo == tipo]
            if not rs:
                continue
            hr = sum(1 for r in rs if r.hit_at_1) / len(rs)
            am = sum(r.mrr for r in rs) / len(rs)
            ap = sum(r.precision_at_3 for r in rs) / len(rs)
            ag = sum(r.gap for r in rs) / len(rs)
            lines.append(
                f"| {tipo} | `{s.model_key}` "
                f"| {hr * 100:.0f}% "
                f"| {am:.3f} "
                f"| {ap:.3f} "
                f"| {ag:+.4f} "
                f"| {len(rs)} |"
            )

    lines += [
        "",
        "---",
        "",
        "## Detalle por caso",
        "",
    ]

    for case in evaluated_cases:
        lines += [
            f"### {case.id} · {case.name}",
            "",
            f"**Tipo:** `{case.tipo}`  ",
            f"**Query:** _{case.query}_  ",
        ]
        if case.note:
            lines.append(f"**Nota:** {case.note}  ")
        lines += [
            f"**Skills relevantes:** {', '.join(f'`{s}`' for s in case.relevant_skills)}  ",
            "",
            "| Modelo | Hit@1 | MRR | P@3 | R@3 | MaxSimRel | MaxSimIrr | Gap | ms/enc |",
            "|--------|-------|-----|-----|-----|-----------|-----------|-----|--------|",
        ]

        for s in summaries:
            match = [r for r in s.results if r.case.id == case.id]
            if not match:
                continue
            r = match[0]
            hit_icon = "✓" if r.hit_at_1 else "✗"
            lines.append(
                f"| `{s.model_key}` "
                f"| {hit_icon} "
                f"| {r.mrr:.3f} "
                f"| {r.precision_at_3:.3f} "
                f"| {r.recall_at_3:.3f} "
                f"| {r.max_sim_relevant:.4f} "
                f"| {r.max_sim_irrelevant:.4f} "
                f"| {r.gap:+.4f} "
                f"| {r.encode_ms:.1f}ms |"
            )

        lines += ["", "**Top-3 recuperado:**", ""]

        for s in summaries:
            match = [r for r in s.results if r.case.id == case.id]
            if not match:
                continue
            r = match[0]
            lines.append(f"*{s.model_key}*")
            lines.append("")
            lines.append("| Rank | Relevancia | Skill | Sim | Repo | Tipo artefacto |")
            lines.append("|------|-----------|-------|-----|------|----------------|")
            for i, rd in enumerate(r.ranking[:3], start=1):
                rel = "✓ REL" if rd.is_relevant else "✗ IRR"
                lines.append(
                    f"| #{i} | {rel} | `{rd.skill}` "
                    f"| {rd.sim:.4f} "
                    f"| {rd.repo} "
                    f"| {rd.artifact_type} |"
                )
            lines.append("")

    lines += [
        "---",
        "",
        "## Conclusión automática",
        "",
    ]
    lines += _auto_conclusion(winner, summaries)
    lines += [
        "",
        "---",
        "*Generado automáticamente por `eval_embeddings.py`*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Markdown guardado en: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

def run() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "═" * W)
    print("  BENCHMARK · Modelos de Embeddings para RAG de RRHH")
    print(f"  {timestamp}")
    print("═" * W)

    # ── Cargar corpus ─────────────────────────────────────────────────────────
    if not PROCESSED_PATH.exists():
        print(f"\n  [ERROR] Fichero no encontrado: {PROCESSED_PATH}")
        print("  Ejecuta el pipeline de extracción antes de lanzar el benchmark.")
        return

    print(f"\n  Cargando corpus desde {PROCESSED_PATH}...")
    records = load_jsonl(PROCESSED_PATH)
    all_records = records + SYNTHETIC_RECORDS
    corpus = build_corpus(all_records)
    print(
        f"  Corpus: {len(corpus)} documentos  "
        f"({len(records)} reales + {len(SYNTHETIC_RECORDS)} sintéticos)"
    )

    corpus_skills = {d["skill"] for d in corpus}
    for case in CASES:
        for skill in case.relevant_skills:
            if skill not in corpus_skills:
                print(f"  [WARN] {case.id}: skill relevante '{skill}' no encontrada en corpus.")

    # ── Precomputar candidatos (una sola vez, mismos para todos los modelos) ──
    rng = random.Random(RANDOM_SEED)
    case_candidates = {
        case.id: select_candidates(case, corpus, rng)
        for case in CASES
    }

    n_skipped = sum(1 for v in case_candidates.values() if not v["relevant_ids"])
    if n_skipped:
        print(f"  [WARN] {n_skipped} casos sin documentos relevantes serán omitidos.")

    print(f"\n  Modelos : {list(MODELS.keys())}")
    print(f"  Casos   : {len(CASES)}")

    # ── Evaluar cada modelo ───────────────────────────────────────────────────
    all_results: list[CaseResult] = []
    summaries: list[ModelSummary] = []

    for model_key, config in MODELS.items():
        results = evaluate_model(model_key, config, corpus, case_candidates)
        all_results.extend(results)
        summaries.append(compute_summary(model_key, results))

    # ── Detalle por caso en consola ───────────────────────────────────────────
    case_by_id: dict = {}
    for s in summaries:
        for r in s.results:
            case_by_id[r.case.id] = r.case
    evaluated_cases = sorted(case_by_id.values(), key=lambda c: c.id)

    for case in evaluated_cases:
        results_by_model = {}
        for mk in MODELS:
            match = [r for r in all_results if r.model_key == mk and r.case.id == case.id]
            if match:
                results_by_model[mk] = match[0]
        if results_by_model:
            print_case_results(case, results_by_model)

    print_summary(summaries)

    # ── Exportar ──────────────────────────────────────────────────────────────
    print("\n  Exportando resultados...")
    export_csv(all_results, OUTPUT_DIR / f"embedding_benchmark_results_{ts_file}.csv")
    export_markdown(
        summaries,
        OUTPUT_DIR / f"embedding_benchmark_summary_{ts_file}.md",
        timestamp,
    )
    print()


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run()

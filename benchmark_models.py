"""
Benchmark de extracción de skills técnicas — comparativa de modelos locales.

Casos de prueba
  C1  true_positive   · Agente multiagente real (LangGraph + tools)
  C2  comment_trap    · LangChain mencionado solo en comentarios/docstrings
  C3  general         · CRUD FastAPI + SQLAlchemy, sin IA
  C4  ambiguous       · Import LangChain sin uso real (código comentado)
  C5  true_positive   · Entrenamiento PyTorch + HuggingFace Trainer
  C6  comment_trap    · Commit de fix genérico (no debería detectar skills IA)
  C7  true_positive   · Pipeline RAG con Anthropic + ChromaDB
  C8  true_positive   · Agente CrewAI multiagente con herramientas

Métricas
  precision    TP / (TP + FP)
  recall       TP / (TP + FN)
  f1           media armónica de P y R
  halluc_rate  skills forbidden detectadas (trampa de comentarios)
  parse_ok     % respuestas JSON válidas (Pydantic)
  latency      segundos por llamada

Salida
  · Consola con detalle por caso y resumen global
  · benchmark_results.csv   — tabla completa por caso y modelo
  · benchmark_summary.md    — tabla Markdown con resumen global + conclusión

Uso
  ollama pull qwen2.5-coder:7b && ollama pull qwen3:4b && ollama pull phi4-mini:3.8b
  python benchmark_models.py
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from pydantic import ValidationError

from src.processing.prompts import DetectionResponse, build_detection_prompt

#  Configuración — edita MODELS según los que tengas en Ollama

MODELS = [
    "qwen2.5-coder:7b",
    "qwen3:4b",
    "phi4-mini:3.8b",
    "granite-code:8b"
]

OLLAMA_URL = "http://localhost:11434/api/generate"
OUTPUT_DIR = Path(".")   # directorio donde se guardan los ficheros de resultados


#  Casos de prueba

@dataclass
class BenchmarkCase:
    id: str
    name: str
    tipo: str           # true_positive | comment_trap | general | ambiguous
    repo: str
    path: str
    artifact_type: str
    content: str
    expected: set = field(default_factory=set)   # skills que SÍ deben aparecer
    forbidden: set = field(default_factory=set)  # skills que NO deben aparecer


CASES: list[BenchmarkCase] = [

    # C1: Agente multiagente real con LangGraph
    BenchmarkCase(
        id="C1",
        name="Agente multiagente real (LangGraph + tools)",
        tipo="true_positive",
        repo="user/multi-agent-system",
        path="src/agents/graph.py",
        artifact_type="file",
        expected={"LangGraph", "LangChain", "OpenAI API"},
        content="""\
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def search_web(query: str) -> str:
    \"\"\"Search the web for information.\"\"\"
    return f"Results for: {query}"

def calculate(expression: str) -> float:
    \"\"\"Evaluate a math expression.\"\"\"
    return eval(expression)

tools = [search_web, calculate]
llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile()
""",
    ),

    # C2: LangChain solo en comentarios — trampa
    BenchmarkCase(
        id="C2",
        name="Trampa: LangChain solo en comentarios/docstrings",
        tipo="comment_trap",
        repo="user/data-processor",
        path="src/pipeline/transform.py",
        artifact_type="file",
        expected={"NumPy", "Pandas"},
        forbidden={"LangChain", "LangGraph", "OpenAI API", "RAG"},
        content="""\
\"\"\"
Data transformation pipeline.

NOTE: We evaluated LangChain and LangGraph for orchestration but decided
against it — too much overhead for our use case. We also considered a RAG
approach with OpenAI API but the latency was unacceptable.
This module handles transformations with plain NumPy and Pandas instead.
\"\"\"

import numpy as np
import pandas as pd


def normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    \"\"\"Z-score normalization. No LangChain, no LLM — just math.\"\"\"
    result = df.copy()
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        result[col] = (df[col] - mean) / std
    return result


def aggregate_by_date(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    # This replaced a LangGraph workflow we prototyped in Q1.
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").resample(freq).sum().reset_index()


def to_numpy_matrix(df: pd.DataFrame) -> np.ndarray:
    return df.select_dtypes(include=[np.number]).to_numpy()
""",
    ),

    # C3: CRUD FastAPI + SQLAlchemy sin IA
    BenchmarkCase(
        id="C3",
        name="CRUD básico FastAPI + SQLAlchemy (sin IA)",
        tipo="general",
        repo="user/inventory-api",
        path="src/api/items.py",
        artifact_type="file",
        expected={"FastAPI", "SQLAlchemy", "Pydantic"},
        forbidden={"LangChain", "LangGraph", "OpenAI API", "PyTorch", "RAG"},
        content="""\
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from src.database import get_db
from src.models import Item

router = APIRouter(prefix="/items", tags=["items"])


class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    quantity: int = Field(..., ge=0)
    price: float = Field(..., gt=0)


class ItemResponse(BaseModel):
    id: int
    name: str
    quantity: int
    price: float
    model_config = {"from_attributes": True}


@router.get("/", response_model=list[ItemResponse])
def list_items(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return db.query(Item).offset(skip).limit(limit).all()


@router.post("/", response_model=ItemResponse, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate, db: Session = Depends(get_db)):
    item = Item(**payload.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int, db: Session = Depends(get_db)):
    item = db.query(Item).filter(Item.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    db.delete(item)
    db.commit()
""",
    ),

    # C4: Import LangChain sin uso real — ambiguo
    BenchmarkCase(
        id="C4",
        name="Ambiguo: import LangChain presente pero sin chain real",
        tipo="ambiguous",
        repo="user/experiment",
        path="notebooks/exploration.py",
        artifact_type="file",
        expected=set(),
        forbidden=set(),
        content="""\
# Quick exploration script — work in progress
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# TODO: build a proper chain here
# For now just testing that the import works

llm = ChatOpenAI(model="gpt-4o")

# This never actually ran — left as reminder
# result = llm.invoke([HumanMessage(content="hello")])
# print(result.content)

print("LLM object created:", llm)
print("Model name:", llm.model_name)
""",
    ),

    # C5: Entrenamiento PyTorch + HuggingFace
    BenchmarkCase(
        id="C5",
        name="Fine-tuning PyTorch + HuggingFace Trainer",
        tipo="true_positive",
        repo="user/llm-finetuning",
        path="src/train.py",
        artifact_type="file",
        expected={"PyTorch", "HuggingFace Transformers"},
        content="""\
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
)

trainer.train()
""",
    ),

    # C6: Commit de fix genérico — trampa (no hay skills IA)
    BenchmarkCase(
        id="C6",
        name="Trampa: commit de fix CSS/HTML sin tecnologías IA",
        tipo="comment_trap",
        repo="user/frontend-app",
        path=None,
        artifact_type="commit",
        expected=set(),
        forbidden={"LangChain", "LangGraph", "OpenAI API", "PyTorch", "RAG",
                   "LlamaIndex", "CrewAI", "FastAPI"},
        content="""\
fix: correct responsive layout on mobile breakpoints

- Adjust grid-template-columns for screens < 768px
- Replace fixed px values with clamp() for font sizes
- Fix overflow hidden on .sidebar cutting off dropdown menus
- Update z-index stacking context for modal overlay
""",
    ),

    # C7: RAG con Anthropic + ChromaDB
    BenchmarkCase(
        id="C7",
        name="Pipeline RAG con Anthropic Claude + ChromaDB",
        tipo="true_positive",
        repo="user/claude-rag",
        path="src/rag/retriever.py",
        artifact_type="file",
        expected={"Anthropic API", "ChromaDB", "Sentence Transformers"},
        content="""\
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

client = anthropic.Anthropic()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection("docs")
embedder = SentenceTransformer("isuruwijesiri/all-MiniLM-L6-v2-code-search-512")


def index_documents(texts: list[str], ids: list[str]) -> None:
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids)


def retrieve(query: str, k: int = 4) -> list[str]:
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    return results["documents"][0]


def answer(question: str) -> str:
    context_chunks = retrieve(question)
    context = "\\n\\n".join(context_chunks)

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Context:\\n{context}\\n\\nQuestion: {question}"
        }],
    )
    return message.content[0].text
""",
    ),

    # C8: CrewAI multiagente con herramientas
    BenchmarkCase(
        id="C8",
        name="Agente CrewAI con roles y herramientas personalizadas",
        tipo="true_positive",
        repo="user/crewai-research",
        path="src/crew.py",
        artifact_type="file",
        expected={"CrewAI", "OpenAI API"},
        content="""\
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for recent information on a topic."

    def _run(self, query: str) -> str:
        # placeholder — integra con SerpAPI / Tavily en producción
        return f"[search results for '{query}']"


researcher = Agent(
    role="Senior Research Analyst",
    goal="Find and summarize the latest developments in AI",
    backstory="Expert in analyzing technical papers and AI news.",
    tools=[WebSearchTool()],
    llm=llm,
    verbose=True,
)

writer = Agent(
    role="Tech Content Writer",
    goal="Write clear, engaging summaries for a technical audience",
    backstory="Experienced technical writer with ML background.",
    llm=llm,
)

research_task = Task(
    description="Research the top 3 AI breakthroughs from the last month.",
    expected_output="Bullet-point summary of 3 breakthroughs with sources.",
    agent=researcher,
)

write_task = Task(
    description="Turn the research into a 300-word blog post.",
    expected_output="A ready-to-publish blog post in Markdown.",
    agent=writer,
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,
    verbose=True,
)

result = crew.kickoff()
""",
    ),
]


#  Cliente Ollama

def call_ollama(model: str, prompt: str) -> tuple[str, float]:
    t0 = time.monotonic()
    r = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.0}},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"], time.monotonic() - t0


def parse_response(text: str) -> Optional[DetectionResponse]:
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    raw = fence.group(1) if fence else raw
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    raw = raw[start:end + 1]
    for candidate in (raw, re.sub(
        r'("(?:[^"\\]|\\.)*")',
        lambda m: m.group(0).replace("\n", "\\n"), raw,
    )):
        try:
            return DetectionResponse.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            pass
    return None


#  Resultado por (modelo, caso)

@dataclass
class CaseResult:
    model: str
    case: BenchmarkCase
    parse_ok: bool
    elapsed: float
    detected: set = field(default_factory=set)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    hallucinations: int = 0

    @property
    def precision(self) -> Optional[float]:
        if not self.case.expected and not self.detected:
            return None
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> Optional[float]:
        if not self.case.expected:
            return None
        return self.tp / len(self.case.expected)

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _match(detected: set, reference: set) -> int:
    d_lower = {s.lower() for s in detected}
    return sum(
        1 for ref in reference
        if any(ref.lower() in d or d in ref.lower() for d in d_lower)
    )


def evaluate(model: str, case: BenchmarkCase) -> CaseResult:
    prompt = build_detection_prompt(
        repo=case.repo,
        path=case.path or case.artifact_type,
        artifact_type=case.artifact_type,
        content=case.content,
    )
    try:
        raw_text, elapsed = call_ollama(model, prompt)
    except Exception as e:
        print(f"      ERROR Ollama: {e}")
        return CaseResult(model=model, case=case, parse_ok=False, elapsed=0)

    parsed = parse_response(raw_text)
    if parsed is None:
        return CaseResult(model=model, case=case, parse_ok=False, elapsed=elapsed)

    detected = {s.skill for s in parsed.skills}
    tp = _match(detected, case.expected)
    fp = max(len(detected) - tp, 0)
    fn = len(case.expected) - tp
    hallucinations = _match(detected, case.forbidden) if case.forbidden else 0

    return CaseResult(
        model=model, case=case, parse_ok=True, elapsed=elapsed,
        detected=detected, tp=tp, fp=fp, fn=fn, hallucinations=hallucinations,
    )


#  Helpers de presentación

W = 96

def sep(c="─"): print(c * W)

def pct(v: Optional[float]) -> str:
    return f"{v*100:5.1f}%" if v is not None else "  n/a "

def bar(v: Optional[float], width: int = 12) -> str:
    if v is None:
        return "─" * width
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


#  Cálculo de métricas globales por modelo

@dataclass
class ModelSummary:
    model: str
    parse_ok: int
    total: int
    avg_precision: Optional[float]
    avg_recall: Optional[float]
    avg_f1: Optional[float]
    total_hallucinations: int
    avg_latency: float
    total_latency: float

    @property
    def score(self) -> float:
        """
        Puntuación compuesta para el ranking final.
          F1 promedio  ×0.50  — calidad de detección
          Precisión    ×0.20  — evitar falsos positivos
          Recall       ×0.20  — no perder skills reales
          Sin alucin.  ×0.10  — bonus por 0 alucinaciones (binario)
        No penalizamos latencia aquí porque depende del hardware.
        """
        f1  = self.avg_f1        or 0.0
        p   = self.avg_precision or 0.0
        r   = self.avg_recall    or 0.0
        hal = 1.0 if self.total_hallucinations == 0 else 0.0
        return round(0.50 * f1 + 0.20 * p + 0.20 * r + 0.10 * hal, 4)


def compute_summary(model: str, results: list[CaseResult]) -> ModelSummary:
    scored = [r for r in results if r.case.tipo != "ambiguous"]
    n = len(scored)
    parse_ok = sum(1 for r in scored if r.parse_ok)

    p_vals = [r.precision for r in scored if r.precision is not None]
    r_vals = [r.recall    for r in scored if r.recall    is not None]
    f_vals = [r.f1        for r in scored if r.f1        is not None]

    return ModelSummary(
        model=model,
        parse_ok=parse_ok,
        total=n,
        avg_precision=sum(p_vals) / len(p_vals) if p_vals else None,
        avg_recall=sum(r_vals)    / len(r_vals) if r_vals else None,
        avg_f1=sum(f_vals)        / len(f_vals) if f_vals else None,
        total_hallucinations=sum(r.hallucinations for r in scored),
        avg_latency=sum(r.elapsed for r in scored) / n if n else 0,
        total_latency=sum(r.elapsed for r in results),  # incluye C4 ambiguo
    )


#  Exportación de resultados

def export_csv(all_results: list[CaseResult], path: Path) -> None:
    """Exporta una fila por (modelo, caso) con todas las métricas."""
    rows = []
    for r in all_results:
        rows.append({
            "modelo":        r.model,
            "caso_id":       r.case.id,
            "caso_nombre":   r.case.name,
            "tipo":          r.case.tipo,
            "parse_ok":      int(r.parse_ok),
            "detectadas":    "; ".join(sorted(r.detected)),
            "tp":            r.tp,
            "fp":            r.fp,
            "fn":            r.fn,
            "precision":     round(r.precision, 4) if r.precision is not None else "",
            "recall":        round(r.recall,    4) if r.recall    is not None else "",
            "f1":            round(r.f1,        4) if r.f1        is not None else "",
            "alucinaciones": r.hallucinations,
            "latencia_s":    round(r.elapsed, 2),
        })

    # Filas de resumen por modelo al final
    from itertools import groupby
    for model, group in groupby(sorted(rows, key=lambda r: r["modelo"]), key=lambda r: r["modelo"]):
        g = list(group)
        total_s = sum(r["latencia_s"] for r in g)
        rows.append({
            "modelo": model, "caso_id": "TOTAL", "caso_nombre": "— resumen modelo —",
            "tipo": "", "parse_ok": sum(r["parse_ok"] for r in g),
            "detectadas": "", "tp": sum(r["tp"] for r in g),
            "fp": sum(r["fp"] for r in g), "fn": sum(r["fn"] for r in g),
            "precision": "", "recall": "", "f1": "",
            "alucinaciones": sum(r["alucinaciones"] for r in g),
            "latencia_s": round(total_s, 2),
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → CSV guardado en: {path}")


def export_markdown(
    summaries: list[ModelSummary],
    all_results: list[CaseResult],
    available: list[str],
    path: Path,
    timestamp: str,
) -> None:
    """Genera un fichero Markdown con tabla de resumen y conclusión."""
    winner = max(summaries, key=lambda s: s.score)
    lines: list[str] = []

    lines += [
        f"# Benchmark · Extracción de skills técnicas",
        f"",
        f"**Fecha:** {timestamp}  ",
        f"**Modelos evaluados:** {', '.join(f'`{m}`' for m in available)}  ",
        f"**Casos de prueba:** {len(CASES)} ({sum(1 for c in CASES if c.tipo != 'ambiguous')} evaluados + 1 ambiguo)  ",
        f"",
        f"---",
        f"",
        f"## Resumen global",
        f"",
        f"| Modelo | Parse OK | Precision | Recall | F1 | Alucinaciones | Media/caso | Tiempo total | Score |",
        f"|--------|----------|-----------|--------|----|---------------|------------|--------------|-------|",
    ]

    def fmt_total(s: ModelSummary) -> str:
        return f"{s.total_latency/60:.1f} min" if s.total_latency >= 60 else f"{s.total_latency:.1f}s"

    for s in sorted(summaries, key=lambda x: x.score, reverse=True):
        medal = "*" if s.model == winner.model else ""
        lines.append(
            f"| {medal}`{s.model}` "
            f"| {s.parse_ok}/{s.total} "
            f"| {pct(s.avg_precision).strip()} "
            f"| {pct(s.avg_recall).strip()} "
            f"| {pct(s.avg_f1).strip()} "
            f"| {'[!] '+str(s.total_hallucinations) if s.total_hallucinations else '[OK] 0'} "
            f"| {s.avg_latency:.1f}s "
            f"| {fmt_total(s)} "
            f"| **{s.score:.2f}** |"
        )

    lines += [
        f"",
        f"> Score = 0.50×F1 + 0.20×Precision + 0.20×Recall + 0.10×(sin alucinaciones)",
        f"",
        f"---",
        f"",
        f"## Detalle por caso",
        f"",
        f"| Caso | Tipo | Modelo | Detectadas | P | R | F1 | Alucin. | s |",
        f"|------|------|--------|------------|---|---|----|---------|---|",
    ]

    for r in all_results:
        if r.case.tipo == "ambiguous":
            skills_str = ", ".join(sorted(r.detected)) if r.detected else "∅"
            lines.append(
                f"| {r.case.id} | ambiguo | `{r.model}` "
                f"| {skills_str} | — | — | — | — | {r.elapsed:.1f} |"
            )
        else:
            ok = "✓" if r.parse_ok else "✗"
            lines.append(
                f"| {r.case.id} | {r.case.tipo} | `{r.model}` "
                f"| {ok} {', '.join(sorted(r.detected)) if r.detected else '∅'} "
                f"| {pct(r.precision).strip()} "
                f"| {pct(r.recall).strip()} "
                f"| {pct(r.f1).strip()} "
                f"| {'[!] '+str(r.hallucinations) if r.hallucinations else '[OK]'} "
                f"| {r.elapsed:.1f} |"
            )

    # Conclusión automática
    runner_up = sorted(summaries, key=lambda x: x.score, reverse=True)
    score_gap = runner_up[0].score - runner_up[1].score if len(runner_up) > 1 else 0

    hal_clean = [s for s in summaries if s.total_hallucinations == 0]
    hal_note = (
        f"Todos los modelos superaron la trampa de comentarios sin alucinaciones."
        if len(hal_clean) == len(summaries)
        else f"Solo {', '.join(f'`{s.model}`' for s in hal_clean)} evitó las alucinaciones."
        if hal_clean
        else "Ningún modelo evitó todas las alucinaciones — revisar el prompt."
    )

    fastest = min(summaries, key=lambda s: s.avg_latency)

    lines += [
        f"",
        f"---",
        f"",
        f"## Conclusión",
        f"",
        f"### Modelo recomendado: `{winner.model}`",
        f"",
        f"Con un score de **{winner.score:.2f}**, `{winner.model}` obtiene el mejor balance "
        f"entre precisión ({pct(winner.avg_precision).strip()}), "
        f"recall ({pct(winner.avg_recall).strip()}) "
        f"y F1 ({pct(winner.avg_f1).strip()}).",
        f"",
        f"{'La ventaja sobre el segundo clasificado es ajustada (Δ={score_gap:.2f}) — ambos son opciones viables.' if score_gap < 0.10 else f'La ventaja sobre el segundo clasificado es clara (Δ={score_gap:.2f}).'}",
        f"",
        f"**Alucinaciones:** {hal_note}",
        f"",
        f"**Latencia:** el modelo más rápido es `{fastest.model}` "
        f"con {fastest.avg_latency:.1f}s de media por llamada "
        f"({fmt_total(fastest)} en total para los {len(CASES)} casos).",
        f"",
        f"### Caso ambiguo (C4)",
        f"",
        f"El caso C4 (import sin uso real) revela la agresividad de cada modelo:",
        f"",
    ]

    for r in all_results:
        if r.case.id == "C4":
            skills_str = ", ".join(sorted(r.detected)) if r.detected else "∅ (conservador)"
            conservador = "[OK] conservador" if not r.detected else "[!] agresivo"
            lines.append(f"- `{r.model}`: {skills_str} — {conservador}")

    lines += [
        f"",
        f"Un modelo conservador (∅ o pocos resultados en C4) es preferible "
        f"para este sistema: reduce el ruido en el perfil técnico del usuario.",
        f"",
        f"---",
        f"*Generado automáticamente por `benchmark_models.py`*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  → Markdown guardado en: {path}")


#  Runner principal

def check_available() -> list[str]:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        tags = {m["name"] for m in resp.json().get("models", [])}
        ok, missing = [], []
        for m in MODELS:
            (ok if m in tags else missing).append(m)
        if missing:
            print(f"  [!] No disponibles: {missing}")
            print(f"    ollama pull " + " && ollama pull ".join(missing))
        return ok
    except Exception as e:
        print(f"  ✗ No se puede conectar a Ollama: {e}")
        return []


def run():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "═" * W)
    print("  BENCHMARK · Extracción de skills técnicas desde código")
    print(f"  {timestamp}")
    print("═" * W)

    available = check_available()
    if not available:
        return

    print(f"\n  Modelos: {available}   Casos: {len(CASES)}\n")

    all_results: list[CaseResult] = []

    # Evaluación por caso
    for case in CASES:
        sep("═")
        tipo_tag = {
            "true_positive": "✓ TRUE POSITIVE",
            "comment_trap":  "[!] TRAMPA",
            "general":       "○ GENERAL",
            "ambiguous":     "? AMBIGUO",
        }.get(case.tipo, case.tipo)
        print(f"  {case.id} · {tipo_tag} · {case.name}")
        if case.expected:
            print(f"  Esperadas  : {sorted(case.expected)}")
        if case.forbidden:
            print(f"  Prohibidas : {sorted(case.forbidden)}")
        sep("·")

        for model in available:
            print(f"\n  [{model}]", flush=True)
            r = evaluate(model, case)
            all_results.append(r)

            if not r.parse_ok:
                print("    ✗ JSON inválido")
                continue

            print(f"    Detectadas : {sorted(r.detected) if r.detected else '∅'}")

            if case.tipo == "ambiguous":
                conserv = "conservador [OK]" if not r.detected else "agresivo [!]"
                print(f"    → {conserv}   ⏱ {r.elapsed:.1f}s")
            else:
                print(f"    Precision  : {pct(r.precision)}  {bar(r.precision)}")
                print(f"    Recall     : {pct(r.recall)}  {bar(r.recall)}")
                print(f"    F1         : {pct(r.f1)}  {bar(r.f1)}")
                if case.forbidden:
                    icon = "[!]" if r.hallucinations else "[OK]"
                    print(f"    {icon} Alucinaciones: {r.hallucinations}")
                print(f"    ⏱ {r.elapsed:.1f}s")
        print()

    # Resumen global
    summaries = [compute_summary(m, [r for r in all_results if r.model == m])
                 for m in available]
    summaries_sorted = sorted(summaries, key=lambda s: s.score, reverse=True)
    winner = summaries_sorted[0]

    sep("═")
    print("  RESUMEN GLOBAL  (excluido C4 ambiguo)")
    sep("═")
    col = 22
    print(f"  {'Modelo':<{col}} {'Parse':<8} {'Precision':<11} {'Recall':<11} {'F1':<11} {'Alucin.':<10} {'Media/caso':<12} {'Total':<10} {'Score'}")
    sep("·")
    for s in summaries_sorted:
        medal = "*  " if s.model == winner.model else "   "
        total_min = f"{s.total_latency/60:.1f}min" if s.total_latency >= 60 else f"{s.total_latency:.1f}s"
        print(
            f"  {medal}{s.model:<{col-3}} {s.parse_ok}/{s.total:<6} "
            f"{pct(s.avg_precision):<11} {pct(s.avg_recall):<11} {pct(s.avg_f1):<11} "
            f"{'[!] '+str(s.total_hallucinations) if s.total_hallucinations else '[OK] 0':<11} "
            f"{s.avg_latency:.1f}s{'':5} {total_min:<10} {s.score:.2f}"
        )
    sep("═")

    # Conclusión en consola
    print()
    print("  CONCLUSIÓN")
    sep("·")
    print(f"  Modelo recomendado : {winner.model}  (score={winner.score:.2f})")
    print(f"  F1 medio           : {pct(winner.avg_f1)}")
    print(f"  Alucinaciones      : {'0 [OK]' if winner.total_hallucinations == 0 else str(winner.total_hallucinations)+' [!]'}")
    fastest = min(summaries, key=lambda s: s.avg_latency)
    total_min = lambda s: f"{s.total_latency/60:.1f}min" if s.total_latency >= 60 else f"{s.total_latency:.1f}s"
    print(f"  Más rápido         : {fastest.model}  ({fastest.avg_latency:.1f}s/llamada · {total_min(fastest)} total)")
    for s in summaries_sorted:
        print(f"    {s.model:<{col}} media={s.avg_latency:.1f}s   total={total_min(s)}")
    print()
    print("  Caso C4 (ambiguo):")
    for r in all_results:
        if r.case.id == "C4":
            tag = "conservador [OK]" if not r.detected else f"agresivo [!]  → {sorted(r.detected)}"
            print(f"    {r.model:<{col}} {tag}")
    sep("═")

    # Exportar resultados
    print("\n  Exportando resultados...")
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_csv(all_results,  OUTPUT_DIR / f"benchmark_results_{ts_file}.csv")
    export_markdown(summaries_sorted, all_results, available,
                    OUTPUT_DIR / f"benchmark_summary_{ts_file}.md", timestamp)
    print()


if __name__ == "__main__":
    run()

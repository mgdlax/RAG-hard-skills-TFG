"""
Ground truth para la evaluacion del sistema RAG de RRHH tecnico.

Contiene 10 consultas de prueba con sus respuestas correctas, disenadas
para reflejar el tipo de busqueda que realizaria un reclutador tecnico
en una empresa centrada en IA aplicada, LLMs y despliegue de modelos.

ESTRUCTURA DE CADA CONSULTA:
  - id            : identificador unico para trazar resultados
  - category      : tipo de consulta (para analisis por categoria)
  - query         : pregunta en lenguaje natural, como la escribiria un reclutador
  - relevant_users: lista de usuarios que son respuesta correcta (relevancia binaria)
  - grades        : relevancia graduada (para NDCG):
                      3 = experto reconocido — proyecto propio dedicado, uso intensivo
                      2 = conocedor avanzado — uso frecuente, integrado en proyectos reales
                      1 = usuario basico    — uso puntual, menciones indirectas
                      0 = irrelevante       — no hace falta incluirlo en el dict

USUARIOS DEL SISTEMA:
  Para evaluar con el ground truth completo, estos 5 usuarios deben estar
  indexados en ChromaDB (ejecutar src/main.py para cada uno):

    chitralputhran — RAG + LangGraph + Streamlit, apps de IA generativa
    ranguy9304     — RAG terminal con LangGraph, sistemas distribuidos, redes neuronales
    naurjhanvi     — Edge AI + RAG + LSTM, despliegue con Docker/Kubernetes/FastAPI
    yldzburhan     — ML/data science, NLP, sistemas de recomendacion, LangChain
    HalemoGPA      — Deep learning, PyTorch, vision por computador, NLP
"""

from typing import Any, Dict, List


#  Dataset de ground truth

GROUND_TRUTH: Dict[str, Any] = {
    "version": "3.0",
    "description": (
        "Conjunto de evaluacion para el sistema RAG de recomendacion tecnica. "
        "10 consultas sobre RAG, LangGraph, despliegue ML, deep learning y NLP."
    ),
    "required_users": [
        "chitralputhran",
        "ranguy9304",
        "naurjhanvi",
        "yldzburhan",
        "HalemoGPA",
    ],
    "queries": [
        # RAG Y RECUPERACION DE INFORMACION
        {
            "id": "q001",
            "category": "rag_systems",
            "query": "Quien recomiendas para construir sistemas RAG de recuperacion de informacion?",
            "relevant_users": ["chitralputhran", "ranguy9304", "naurjhanvi"],
            "grades": {
                "chitralputhran": 3,  # Proyecto Advanced-RAG-LangGraph dedicado a RAG con ChromaDB
                "ranguy9304": 3,      # Proyecto LangGraphRAG: sistema RAG terminal con vector DB
                "naurjhanvi": 2,      # RAG con LangChain + FAISS en Argus_Echo para ICS
                "yldzburhan": 1,      # Usa RAG con ChromaDB en sus proyectos de ML
            },
        },
        {
            "id": "q002",
            "category": "rag_systems",
            "query": "Necesito a alguien con experiencia en embeddings semanticos y busqueda vectorial",
            "relevant_users": ["chitralputhran", "ranguy9304", "naurjhanvi"],
            "grades": {
                "chitralputhran": 3,  # ChromaDB como vector store en Advanced-RAG-LangGraph
                "ranguy9304": 2,      # Vector DB en LangGraphRAG con message caching
                "naurjhanvi": 2,      # FAISS para busqueda vectorial en Argus_Echo
                "yldzburhan": 1,      # ChromaDB en proyectos de ML/RAG
            },
        },
        # ORQUESTACION CON LANGGRAPH
        {
            "id": "q003",
            "category": "orchestration",
            "query": "Quien tiene mas experiencia con LangGraph para construir flujos de agentes?",
            "relevant_users": ["chitralputhran", "ranguy9304"],
            "grades": {
                "chitralputhran": 3,  # Advanced-RAG-LangGraph: usa LangGraph como backbone principal
                "ranguy9304": 3,      # LangGraphRAG: sistema RAG construido sobre LangGraph
                "naurjhanvi": 1,      # Posible uso de LangChain/LangGraph en pipelines de agentes
            },
        },
        {
            "id": "q004",
            "category": "orchestration",
            "query": "Quien sabe integrar LangChain para construir aplicaciones con LLMs?",
            "relevant_users": ["chitralputhran", "naurjhanvi", "yldzburhan"],
            "grades": {
                "chitralputhran": 3,  # LangChain en Advanced-RAG-LangGraph y Recipe-AI
                "naurjhanvi": 2,      # LangChain + Groq en Argus_Echo para RAG industrial
                "yldzburhan": 2,      # LangChain en proyectos de IA generativa y RAG
                "ranguy9304": 1,      # LangGraph es parte del ecosistema LangChain
            },
        },
        # DESPLIEGUE Y PRODUCCION
        {
            "id": "q005",
            "category": "deployment",
            "query": "Quien tiene experiencia desplegando modelos de ML en produccion con Docker y Kubernetes?",
            "relevant_users": ["naurjhanvi"],
            "grades": {
                "naurjhanvi": 3,   # Argus_Echo desplegado con Docker y Google Kubernetes Engine
                "yldzburhan": 2,   # Infraestructura cloud (AWS, Azure, GCP) y contenedores
                "HalemoGPA": 1,    # Proyectos con algun nivel de despliegue
            },
        },
        {
            "id": "q006",
            "category": "deployment",
            "query": "Quien sabe exponer modelos de IA como API REST con FastAPI?",
            "relevant_users": ["naurjhanvi", "chitralputhran"],
            "grades": {
                "naurjhanvi": 3,      # AgriPredictAI usa FastAPI como backend de los modelos ML
                "chitralputhran": 2,  # Aplicaciones Streamlit con backend de servicios IA
                "yldzburhan": 1,      # Django/Flask/Streamlit en sus proyectos de ML
            },
        },
        {
            "id": "q007",
            "category": "deployment",
            "query": "Quien sabe desarrollar aplicaciones de IA interactivas con Streamlit?",
            "relevant_users": ["chitralputhran", "yldzburhan"],
            "grades": {
                "chitralputhran": 3,  # Multiples apps Streamlit: Advanced-RAG, Recipe-AI, etc.
                "yldzburhan": 2,      # Usa Streamlit en proyectos de ML y visualizacion
                "naurjhanvi": 1,      # Interfaces web en algunos proyectos
            },
        },
        # DEEP LEARNING Y VISION
        {
            "id": "q008",
            "category": "deep_learning",
            "query": "Quien domina deep learning y vision por computador con PyTorch?",
            "relevant_users": ["HalemoGPA", "naurjhanvi"],
            "grades": {
                "HalemoGPA": 3,    # BrainMRI-Tumor-Classifier-Pytorch: clasificacion medica con PyTorch
                "naurjhanvi": 2,   # CNN para deteccion de enfermedades en AgriPredictAI y LSTM en Argus
                "yldzburhan": 1,   # Modelos ML con redes neuronales en competiciones
            },
        },
        # NLP Y ANALISIS DE TEXTO
        {
            "id": "q009",
            "category": "nlp",
            "query": "Quien tiene mas experiencia en procesamiento de lenguaje natural y analisis de texto?",
            "relevant_users": ["yldzburhan", "HalemoGPA"],
            "grades": {
                "yldzburhan": 3,   # NLP en turco: sentiment analysis, NER, traduccion de recursos
                "HalemoGPA": 2,    # Finalista competicion NLP arabe, proyectos de texto
                "naurjhanvi": 1,   # LLM grounding y RAG implican procesamiento de texto
            },
        },
        # CIENCIA DE DATOS Y ML CLASICO
        {
            "id": "q010",
            "category": "data_science",
            "query": "Quien recomendarías para un proyecto de sistemas de recomendacion y analisis de datos de clientes?",
            "relevant_users": ["yldzburhan"],
            "grades": {
                "yldzburhan": 3,   # Proyecto Recommendation-Systems y CRM-Analytics-RFM dedicados
                "HalemoGPA": 1,    # Proyecto Machine-Learning-Project con modelos clasicos
                "ranguy9304": 1,   # Knowledge-Based-Agent como sistema de recomendacion con RAG
            },
        },
    ],
}


#  Helpers de acceso

def get_all_queries() -> List[Dict[str, Any]]:
    """Devuelve todas las consultas del ground truth."""
    return GROUND_TRUTH["queries"]


def get_required_users() -> List[str]:
    """Lista de usuarios que deben estar indexados para la evaluacion completa."""
    return GROUND_TRUTH["required_users"]

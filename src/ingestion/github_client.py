"""
Cliente autenticado para la API REST de GitHub.

Usa un token personal (PAT) desde el .env para acceder a los 5000
requests/hora del tier autenticado. Sin token, la API publica limita
a 60 requests/hora, insuficiente para el pipeline completo.
"""

import os
from dotenv import load_dotenv
from github import Github


def get_github_client() -> Github:
    """
    Devuelve un cliente PyGithub autenticado.
    Lee GITHUB_TOKEN del .env o de las variables de entorno del sistema.
    Lanza ValueError si el token no esta configurado.
    """
    load_dotenv()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "No se encontro GITHUB_TOKEN. "
            "Crea un .env con: GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx"
        )

    return Github(token)
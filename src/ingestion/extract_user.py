"""
Extraccion de evidencias de todos los repositorios de un usuario GitHub.

Selecciona los repositorios mas relevantes (por stars + actividad reciente),
descarta forks y repos archivados, y delega la extraccion de artefactos
a extract_repo.py (ficheros, commits, issues, PRs).
"""

import time
from typing import List

from github.NamedUser import NamedUser

from src.ingestion.extract_repo import (
    walk_repository_files,
    extract_commits,
    extract_issues,
    extract_pull_requests,
)
from src.ingestion.schemas import GitHubEvidence
from src.utils.logger import get_logger

log = get_logger(__name__)


def is_candidate_repo(repo) -> bool:
    """Descarta forks y repositorios archivados."""
    if repo.fork:
        return False
    if repo.archived:
        return False
    return True


def extract_user_profile_evidences(
    user: NamedUser,
    max_repos: int = 4,
    max_files_per_repo: int = 40,
    max_commits_per_repo: int = 20,
    max_issues_per_repo: int = 10,
    max_prs_per_repo: int = 10,
) -> List[GitHubEvidence]:
    """
    Extrae evidencias tecnicas de los mejores repositorios del usuario.

    Ordena los repos por (stars, actividad reciente) y selecciona los
    max_repos primeros. Por cada uno descarga ficheros, commits, issues
    y PRs hasta los limites indicados.
    """
    log.info(f"Analizando usuario: {user.login}")

    all_repos = list(user.get_repos())
    candidate_repos = [r for r in all_repos if is_candidate_repo(r)]

    log.info(f"Repositorios publicos: {len(all_repos)}")
    log.info(f"Candidatos (sin forks ni archivados): {len(candidate_repos)}")
    log.debug(
        f"Repos descartados: "
        f"{[r.full_name for r in all_repos if not is_candidate_repo(r)]}"
    )

    repos_sorted = sorted(
        candidate_repos,
        key=lambda r: (r.stargazers_count, r.pushed_at.timestamp() if r.pushed_at else 0),
        reverse=True,
    )

    selected_repos = repos_sorted[:max_repos]
    log.info(
        f"Seleccionados (top {max_repos} por stars + actividad reciente):"
    )
    for r in selected_repos:
        log.info(
            f"  • {r.full_name}  "
            f"[stars: {r.stargazers_count}, último push: {r.pushed_at}]"
        )

    evidences = []
    t_total = time.monotonic()

    for repo in selected_repos:
        t_repo = time.monotonic()
        log.info(f"Extrayendo {repo.full_name} ...")

        files = walk_repository_files(
            repo, username=user.login, max_files=max_files_per_repo
        )
        commits = extract_commits(
            repo, username=user.login, max_commits=max_commits_per_repo
        )
        issues = extract_issues(
            repo, username=user.login, max_issues=max_issues_per_repo
        )
        prs = extract_pull_requests(
            repo, username=user.login, max_prs=max_prs_per_repo
        )

        elapsed_repo = time.monotonic() - t_repo
        log.info(
            f"  {repo.full_name}: "
            f"{len(files)} ficheros, {len(commits)} commits, "
            f"{len(issues)} issues, {len(prs)} PRs  "
            f"({elapsed_repo:.1f}s)"
        )

        evidences.extend(files)
        evidences.extend(commits)
        evidences.extend(issues)
        evidences.extend(prs)

    elapsed_total = time.monotonic() - t_total
    log.info(
        f"Extraccion completada: {len(evidences)} evidencias "
        f"de {len(selected_repos)} repositorios en {elapsed_total:.1f}s"
    )
    return evidences

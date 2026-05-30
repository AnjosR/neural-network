"""Carregamento e parsing do conjunto de dados de treinamento."""

from __future__ import annotations

from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "Desafio_Cerebral_-_Dados_de_Treinamento.txt"
)


def load_dataset(path: Path | str = DATA_FILE):
    """Lê o arquivo de treinamento e retorna as entradas e a saída desejada.

    O arquivo possui um cabeçalho (``x1 x2 x3 d``) seguido das amostras.

    TODO: implementar a leitura.
    """
    raise NotImplementedError

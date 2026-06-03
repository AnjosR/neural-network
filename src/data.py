from __future__ import annotations

from pathlib import Path

DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "desafio_cerebral.txt"
)


def carregar_dataset(path: Path | str = DATA_FILE):
    """Lê o arquivo de treinamento e retorna as entradas e a saída desejada.\n"""

    try:
        with open(path, "r", encoding="utf-8") as arquivo:
            linhas = arquivo.readlines()
    except FileNotFoundError:
        print(f"Erro: O arquivo {path} não foi encontrado.")

    amostras: list[list[float]] = []
    for dados in linhas[1:]:
        itens = [float(valor) for valor in dados.split()]
        amostras.append(itens)

    return amostras
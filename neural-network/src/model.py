from __future__ import annotations

LIMIAR = -1.0

def preparar_entrada(entradas: list[float]) -> list[float]:
    """Adiciona a Limiar (-1) à lista de entradas."""
    return [LIMIAR, *entradas]


def potencial_ativacao(pesos: list[float], entradas: list[float]) -> float:
    """Multiplica os pesos com as suas respectivas entradas, e retorna a soma das multiplicações."""
    return sum(peso * entrada for peso, entrada in zip(pesos, entradas))


def funcao_ativacao(potencial: float) -> float:
    """Retorna +1 se potencial ≥ 0, caso contrário -1."""
    return 1.0 if potencial >= 0 else -1.0


def prever(pesos: list[float], entradas: list[float]) -> float:
    """Saída do neurônio (+1 ou -1) para uma amostra já preparada com limiar."""
    return funcao_ativacao(potencial_ativacao(pesos, entradas))
"""Implementação do neurônio (Perceptron)."""

from __future__ import annotations

import random

LIMIAR = -1.0
TAXA_APRENDIZAGEM = 0.01
LIMITE_REAJUSTES = 5


def funcao_ativacao(soma: float) -> float:
    """Função de ativação (degrau bipolar): retorna 1 ou -1."""
    return 1.0 if soma >= 0 else -1.0


def neuronio(dados: list[float]) -> dict:
    """Processa uma lista de floats com um neurônio (Perceptron).\n
    Retorna um dicionário com o resultado, o nº de reajustes, os pesos finais
    e o histórico de cada tentativa.
    """
    
    entradas = dados[:-1]
    esperado = dados[-1]   

    pesos = [random.uniform(0, 1) for _ in entradas]

    reajustes = 0

    while True:
        soma = sum(x * w for x, w in zip(entradas, pesos)) + LIMIAR
        y = funcao_ativacao(soma)

        if y == esperado:
            return {
                "resultado": y,
                "esperado": esperado,
                "reajustes": reajustes,
                "pesos": pesos,
                "convergiu": True,
            } 
        else:
            pesos = [
                peso + TAXA_APRENDIZAGEM * (esperado - y) * x
                for peso, x in zip(pesos, entradas)
            ]
            reajustes += 1

        if reajustes >= LIMITE_REAJUSTES:
            return {
                "resultado": y,
                "esperado": esperado,
                "reajustes": reajustes,
                "pesos": pesos,
                "convergiu": False,
            }
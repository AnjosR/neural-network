"""Treinamento do Perceptron (aprendizado supervisionado, regra de Hebb).

O treinamento percorre o conjunto de amostras em épocas sucessivas. A cada
época, todas as amostras são apresentadas; sempre que a saída prevista difere
da desejada, os pesos são ajustados na direção do erro. A rede é considerada
treinada quando uma época inteira ocorre sem nenhum erro (convergência).
"""

from __future__ import annotations

import random
from collections import namedtuple

from model import prever, preparar_entrada

TAXA_APRENDIZAGEM = 0.01
MAX_EPOCAS = 1000

ResultadoTreinamento = namedtuple(
    "ResultadoTreinamento", ["pesos_iniciais", "pesos", "epocas", "convergiu"]
)


def preparar_conjunto(
    amostras: list[list[float]],
) -> list[tuple[list[float], float]]:
    """Separa cada amostra em (entradas com bias, saída desejada)."""
    return [
        (preparar_entrada(amostra[:-1]), amostra[-1])
        for amostra in amostras
    ]


def inicializar_pesos(quantidade: int) -> list[float]:
    """Inicia o vetor de pesos com valores aleatórios pequenos."""
    return [random.uniform(0.0, 1.0) for _ in range(quantidade)]


def ajustar_pesos(
    pesos: list[float], entradas: list[float], erro: float, taxa: float
) -> list[float]:
    """Aplica a regra de Hebb: w ← w + η·(d - y)·x."""
    return [
        peso + taxa * erro * entrada
        for peso, entrada in zip(pesos, entradas)
    ]


def executar_epoca(
    pesos: list[float], conjunto: list[tuple[list[float], float]], taxa: float
) -> tuple[list[float], bool]:
    """Apresenta todas as amostras uma vez; devolve os pesos e se houve erro."""
    houve_erro = False
    for entradas, esperado in conjunto:
        erro = esperado - prever(pesos, entradas)
        if erro != 0:
            pesos = ajustar_pesos(pesos, entradas, erro, taxa)
            houve_erro = True
    return pesos, houve_erro


def treinar(
    amostras: list[list[float]],
    taxa_aprendizagem: float = TAXA_APRENDIZAGEM,
    max_epocas: int = MAX_EPOCAS,
) -> ResultadoTreinamento:
    """Treina o Perceptron e devolve pesos iniciais, finais, nº de épocas e
    se houve convergência."""
    
    conjunto = preparar_conjunto(amostras)
    pesos_iniciais = inicializar_pesos(len(conjunto[0][0]))
    pesos = list(pesos_iniciais)

    for epoca in range(1, max_epocas + 1):
        pesos, houve_erro = executar_epoca(pesos, conjunto, taxa_aprendizagem)
        if not houve_erro:
            return ResultadoTreinamento(pesos_iniciais, pesos, epoca, convergiu=True)

    return ResultadoTreinamento(pesos_iniciais, pesos, max_epocas, convergiu=False)
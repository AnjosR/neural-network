"""Ponto de entrada do projeto."""

from __future__ import annotations

from data import load_dataset
from model import neuronio

def main():
    amostras = load_dataset()

    for numero, amostra in enumerate(amostras, start=1):
        resultado = neuronio(amostra)
        print(
            f"amostra {numero:2d} | "
            f"esperado={resultado['esperado']:+.0f} | "
            f"resultado={resultado['resultado']:+.0f} | "
            f"reajustes={resultado['reajustes']}"
        )

if __name__ == "__main__":
    main()
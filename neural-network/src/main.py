from __future__ import annotations

from data import carregar_dataset
from model import prever, preparar_entrada
from train import treinar


def formatar_pesos(pesos: list[float]) -> str:
    """Mostra o vetor de pesos com o limiar (θ) destacado."""
    limiar, *sinapticos = pesos
    sinapticos_fmt = ", ".join(f"{peso:+.3f}" for peso in sinapticos)
    return f"θ={limiar:+.3f} | w=[{sinapticos_fmt}]"


def main() -> None:
    amostras = carregar_dataset()
    resultado = treinar(amostras)

    if resultado.convergiu:
        print(f"Rede treinada em {resultado.epocas} época(s).")
    else:
        print(f"A rede não convergiu após {resultado.epocas} épocas.")

    print(f"Pesos finais: {formatar_pesos(resultado.pesos)}\n")


if __name__ == "__main__":
    main()
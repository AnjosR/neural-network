from __future__ import annotations

from data import carregar_dataset
from model import preparar_entrada, prever
from train import treinar


def formatar_pesos(pesos: list[float]) -> str:
    """Mostra o vetor de pesos com o limiar (θ) destacado."""
    limiar, *sinapticos = pesos
    sinapticos_fmt = ", ".join(f"{peso:+.3f}" for peso in sinapticos)
    return f"θ={limiar:+.3f} | w=[{sinapticos_fmt}]"

CLASSES = {1.0: "P1", -1.0: "P2"}

# Amostras da fase de operação (classificação automática), conforme o Projeto Prático.
AMOSTRAS_TESTE = [
    [-0.3665, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548],
]


def main() -> None:

    redes = []
    for _ in range(5):
        amostras = carregar_dataset()
        resultado = treinar(amostras)

        if resultado.convergiu:
            print(f"Rede treinada em {resultado.epocas} época(s).")
        else:
            print(f"A rede não convergiu após {resultado.epocas} épocas.")

        print(f"Pesos iniciais: {formatar_pesos(resultado.pesos_iniciais)}")
        print(f"Pesos finais: {formatar_pesos(resultado.pesos)}\n")
        redes.append(resultado.pesos)

    print("Classificação das amostras:")
    for i, amostra in enumerate(AMOSTRAS_TESTE, start=1):
        entradas = preparar_entrada(amostra)
        tipos = [CLASSES[prever(pesos, entradas)] for pesos in redes]
        print(f"Amostra {i}: {', '.join(tipos)}")


if __name__ == "__main__":
    main()
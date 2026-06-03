# Neural Network — Desafio Cerebral

Implementação de um **Perceptron** treinado por aprendizado supervisionado
com a **regra de Hebb**, a partir do conjunto de dados em
`docs/desafio_cerebral.txt`.

O dataset possui 4 colunas: três entradas (`x1`, `x2`, `x3`) e a saída
desejada `d` (com valores `-1.0` ou `+1.0`).

```
        x1        x2       x3          d
   -0.6508    0.1097    4.0009   -1.0000
   -1.4492    0.8896    4.4005   -1.0000
    ...
```

## Como funciona

- A cada amostra, um **limiar** (`θ = -1.0`) é adicionado às entradas e o
  potencial de ativação é calculado pela soma ponderada `Σ wᵢ·xᵢ`.
- A **função de ativação** (degrau bipolar) retorna `+1` se o potencial for
  `≥ 0`, caso contrário `-1`.
- Sempre que a saída prevista difere da desejada, os pesos são ajustados pela
  regra de Hebb: `w ← w + η·(d - y)·x`, com taxa de aprendizagem `η = 0.01`.
- O treinamento percorre as amostras em épocas sucessivas e converge quando
  uma época inteira ocorre sem nenhum erro (limite de `1000` épocas).

## Estrutura

```
neural-network/
├── docs/                       # dados e documentação
│   ├── desafio_cerebral.txt    # dataset de treinamento
│   └── formula_hebb.txt        # nota sobre a regra de ajuste
├── src/
│   ├── __init__.py
│   ├── data.py                 # carregamento/parse do dataset
│   ├── model.py                # neurônio: potencial e ativação
│   ├── train.py                # laço de treinamento (regra de Hebb)
│   └── main.py                 # ponto de entrada
├── tests/
├── requirements.txt
└── README.md
```

## Como usar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

O `main.py` executa o treinamento 5 vezes (cada uma com pesos iniciais
aleatórios) e imprime, para cada execução, o número de épocas até a
convergência e o vetor de pesos final:

```
Rede treinada em 411 época(s).
Pesos finais: θ=-3.075 | w=[+1.530, +2.512, -0.707]
```

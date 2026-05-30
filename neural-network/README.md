# Neural Network — Desafio Cerebral

Projeto para treinamento de uma rede neural a partir do conjunto de dados em
`docs/Desafio_Cerebral_-_Dados_de_Treinamento.txt`.

O dataset possui 4 colunas: três entradas (`x1`, `x2`, `x3`) e a saída desejada
`d` (com valores `-1.0` ou `+1.0`).

## Estrutura

```
neural-network/
├── data/                 # dados processados (gerados em runtime)
├── docs/                 # dados e documentação
│   └── Desafio_Cerebral_-_Dados_de_Treinamento.txt
├── src/
│   └── neural_network/
│       ├── __init__.py
│       ├── data.py       # carregamento/parse do dataset
│       ├── model.py      # implementação da rede
│       └── train.py      # laço de treinamento
├── tests/
├── main.py               # ponto de entrada
├── requirements.txt
└── README.md
```

## Como usar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

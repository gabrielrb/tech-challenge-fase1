# Tech Challenge Fase 1 — Apoio ao diagnóstico com Machine Learning

Pós-Tech **IA para Devs** (FIAP) · Fase 1

| Notebook | Abrir |
| --- | --- |
| Parte A: câncer de mama | [![Abrir a Parte A no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabrielrb/tech-challenge-fase1/blob/main/diagnostico_cancer_mama.ipynb) |
| Parte B (EXTRA): pneumonia | [![Abrir a Parte B no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabrielrb/tech-challenge-fase1/blob/main/diagnostico_pneumonia.ipynb) |

**Relatório técnico:** [relatorio_tecnico.pdf](relatorio_tecnico.pdf)

> **Aviso:** este é um projeto acadêmico. Os modelos apoiam a triagem de exames e não substituem o diagnóstico. A decisão final é sempre do médico.

## Sumário

- [O projeto](#o-projeto)
- [Resultados](#resultados)
- [Como foi feito](#como-foi-feito)
- [Como executar](#como-executar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Bases de dados](#bases-de-dados)
- [Tecnologias](#tecnologias)
- [Limitações](#limitações)
- [Integrantes](#integrantes)

## O projeto

Um hospital universitário quer um sistema de apoio ao diagnóstico para acelerar a triagem de exames. O sistema classifica cada exame como "doente" ou "não doente" e aponta os casos suspeitos para o médico olhar primeiro.

| Parte | Problema | Dados | Modelos comparados |
| --- | --- | --- | --- |
| A | câncer de mama: maligno ou benigno | 569 pacientes, 30 medidas do núcleo das células (tabela) | KNN, SVM, Árvore de Decisão e Random Forest |
| B (EXTRA) | pneumonia no raio-X de tórax | 5.856 radiografias de crianças | 2 redes neurais convolucionais (CNN) e PCA + Random Forest |

**Métrica principal: recall da classe doente.** O erro mais grave é o falso negativo, ou seja, dizer que um doente está saudável. O recall mede quantos doentes o modelo encontra. A precisão, o F1, a acurácia e a AUC ficam como métricas de apoio.

**Uso na prática:** o modelo ordena a fila de exames pela chance de doença. O médico continua revendo todos os exames e decide.

## Resultados

Resultados no conjunto de teste, que só foi usado no fim, depois da escolha do modelo. Os números da Parte A foram gerados com o scikit-learn 1.6.1: em outra versão, o Random Forest pode mudar um pouco.

| Parte | Problema | Modelo final | Recall da classe doente | Outras métricas |
| --- | --- | --- | --- | --- |
| A | câncer de mama (tabela) | Random Forest | 0,93 (39 de 42 cânceres) | acurácia 0,97 · precisão 1,00 · F1 0,96 · AUC 0,994 |
| B | pneumonia (raio-X) | CNN 2 | 0,99 (385 de 390 pneumonias) | recall de NORMAL 0,41 · AUC 0,896 |

### Parte A: câncer de mama

Na validação cruzada (5 partes, só com o treino), o SVM e o Random Forest ficaram à frente. Os dois foram ajustados com GridSearchCV, pelo recall.

| Modelo | Recall na validação | Desvio entre as rodadas |
| --- | --- | --- |
| SVM ajustado | 0,959 | 0,015 |
| Random Forest ajustado | 0,938 | 0,042 |
| SVM (padrão) | 0,942 | 0,035 |
| Random Forest (padrão) | 0,933 | 0,051 |
| Árvore de Decisão | 0,922 | 0,045 |
| KNN | 0,911 | 0,041 |

O Random Forest foi o escolhido. O SVM teve recall um pouco maior, mas a diferença de 0,021 dá menos de 1 câncer por rodada e é menor que o desvio do próprio Random Forest (0,042). O Random Forest mostra de forma direta quais medidas pesam na decisão, o que ajuda o médico a confiar no modelo. No teste, ele acertou 111 de 114 pacientes (acurácia de 0,97), sem nenhum alarme falso. Os 3 erros são cânceres classificados como benignos.

<p align="center">
  <img src="figuras/matriz_confusao.png" alt="Matriz de confusão do Random Forest no teste" width="45%">
  <img src="figuras/feature_importance.png" alt="As 10 medidas mais importantes do Random Forest" width="45%">
</p>

As medidas que mais pesam formam dois grupos: o tamanho do núcleo da célula (perímetro, área e raio) e a concavidade (pontos côncavos). Isso combina com a análise de correlação e faz sentido médico.

O **SHAP** completa a leitura. A importância das variáveis dá um número só para o modelo todo; o SHAP mostra a direção de cada medida e explica a decisão de cada paciente. As 5 medidas que mais pesam pelo SHAP são as mesmas 5, e valores altos delas (pontos vermelhos) empurram para maligno. Ele também explica um dos 3 cânceres perdidos: as medidas desse paciente estavam mais perto das de um tumor benigno.

<p align="center">
  <img src="figuras/shap_resumo.png" alt="SHAP: cada ponto é um paciente; à direita, a medida empurra para maligno" width="70%">
</p>

**Limiar de decisão: cânceres perdidos × alarmes falsos.** O modelo dá uma chance de câncer para cada paciente, e o corte padrão é 0,5. Baixar o corte pega mais cânceres, mas gera mais alarmes falsos. Cada corte foi escolhido na validação cruzada, só com o treino, e depois conferido no teste:

| Meta de recall na validação | Limiar | Cânceres perdidos no teste | Alarmes falsos no teste |
| --- | --- | --- | --- |
| padrão | 0,5 | 3 de 42 | 0 de 72 |
| 98% | 0,21 | 2 de 42 | 6 de 72 |
| 99% | 0,03 | 0 de 42 | 23 de 72 |
| 100% | 0,007 | 0 de 42 | 44 de 72 |

Os últimos cânceres custam muitos alarmes falsos, porque alguns têm medidas de tumor benigno. Por isso, a escolha do limiar fica com a equipe médica. Os resultados acima usam o limiar padrão.

<p align="center">
  <img src="figuras/limiar_tradeoff.png" alt="Troca entre cânceres perdidos e alarmes falsos conforme o limiar" width="60%">
</p>

### Parte B: pneumonia

A validação foi separada por paciente: as radiografias de uma mesma criança ficam todas no treino ou todas na validação. Assim, a rede não é avaliada com crianças que já viu.

O notebook compara as 3 soluções na validação e escolhe sozinho o modelo final, antes de olhar o teste. A regra: descarta os modelos com recall de NORMAL abaixo de 0,80 e, entre os demais, fica com o maior recall de PNEUMONIA. Se dois ficarem a até 0,02 um do outro, vence o de maior recall de NORMAL.

A regra escolheu a CNN 2. No teste, ela encontrou 385 das 390 pneumonias, mas acertou só 41% dos exames normais (139 alarmes falsos). Na validação, esse acerto era de 95%: a validação não mostrou a queda. Por isso, o modelo de imagens ainda precisa melhorar antes de qualquer uso.

<p align="center">
  <img src="figuras/pneumonia_matriz.png" alt="Matriz de confusão do modelo final da Parte B no teste" width="45%">
  <img src="figuras/pneumonia_oclusao.png" alt="Mapa de oclusão: as áreas que mais pesaram na decisão" width="45%">
</p>

O mapa de oclusão cobre um pedaço da imagem por vez e mede quanto a certeza de pneumonia cai. Cada raio-X (em cima) tem o seu mapa embaixo: quanto mais forte o vermelho, mais aquele quadrado pesou para PNEUMONIA.

## Como foi feito

| Etapa | Parte A: câncer de mama | Parte B: pneumonia |
| --- | --- | --- |
| Exploração | estatísticas descritivas, contagem das classes, histogramas, boxplot e correlação | contagem das classes, exemplos, imagem média e brilho de cada classe |
| Pré-processamento | remove `id` e `Unnamed: 32`; alvo com LabelEncoder (maligno = 1); padronização dentro de um Pipeline | tons de cinza, 128 × 128 pixels, valores entre 0 e 1; pneumonia = 1 |
| Divisão | 80% treino e 20% teste, estratificado; validação cruzada KFold de 5 partes no treino | pasta `test` como teste (624 imagens); validação de cerca de 20% tirada do treino, separada por paciente e estratificada |
| Modelos | KNN, SVM, Árvore e Random Forest; teste com PCA; GridSearchCV nos 2 melhores | CNN 1 (3 convoluções); CNN 2 (128 filtros, um pooling a mais e Dropout); PCA + Random Forest |
| Avaliação | matriz de confusão, recall, precisão, F1, curva ROC e AUC | recall das 2 classes, matriz de confusão, curva ROC e AUC |
| Interpretação | importância das variáveis, SHAP, uma árvore de 3 níveis desenhada e a troca do limiar de decisão | grade de acertos e erros e mapa de oclusão |

Os detalhes e as justificativas de cada escolha estão no [relatório técnico](relatorio_tecnico.pdf) e nos textos dos próprios notebooks.

## Como executar

### Opção 1: Google Colab (recomendado)

Não precisa instalar nada.

1. Clique no botão **Open in Colab** da parte que quer rodar, na tabela do topo desta página. Se aparecer o aviso de que o notebook não foi criado pelo Google, clique em **Executar mesmo assim**.
2. Na Parte B, ligue a GPU: **Ambiente de execução → Alterar o tipo de ambiente de execução → T4 GPU → Salvar** (em inglês: *Runtime → Change runtime type → T4 GPU → Save*).
3. Rode tudo: **Ambiente de execução → Executar tudo** (em inglês: *Runtime → Run all*).

| Parte | Tempo aproximado | Dados |
| --- | --- | --- |
| A | poucos minutos | o `data.csv` é lido deste repositório |
| B | 15 a 30 minutos, com GPU | o notebook baixa a base do Kaggle (cerca de 2,5 GB) |

### Opção 2: Docker

Requer o [Docker](https://docs.docker.com/get-docker/) instalado. O `docker build` leva alguns minutos, porque instala o TensorFlow.

```bash
git clone https://github.com/gabrielrb/tech-challenge-fase1.git
cd tech-challenge-fase1
docker build -t tech-challenge .
docker run --rm -p 127.0.0.1:8888:8888 tech-challenge
```

Abra <http://localhost:8888>, entre no notebook e use **Run → Run All Cells**. Para encerrar, aperte `Ctrl + C` no terminal.

- O Jupyter do contêiner abre sem senha. Por isso, o `127.0.0.1` deixa a porta acessível só no seu computador.
- O que você muda no notebook fica dentro do contêiner e some quando ele é encerrado.
- No Docker, a Parte B roda na CPU. Num computador com 2 núcleos, sem GPU, ela inteira levou cerca de 17 minutos, já contando o download da base.
- Sem GPU, os números das CNNs mudam um pouco em relação ao Colab. No nosso teste no Docker, a CNN 2 encontrou 388 das 390 pneumonias, e o recall de NORMAL foi 0,37 (no Colab: 385 e 0,41). A escolha do modelo e a conclusão são as mesmas.

**Docker no navegador (GitHub Codespaces):** no GitHub, clique em **Code → Codespaces → Create codespace on main**. O repositório já vem aberto. No terminal do Codespace, rode só os dois últimos comandos acima (`docker build` e `docker run`). Quando aparecer o aviso da porta 8888, clique em **Open in Browser**.

### Opção 3: no computador, sem Docker

Requer Python 3.13, a mesma versão do Colab e do Dockerfile. Em Mac com processador Intel, o TensorFlow 2.21 não instala: use o Colab ou o Docker.

Linux ou macOS:

```bash
git clone https://github.com/gabrielrb/tech-challenge-fase1.git
cd tech-challenge-fase1
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Windows (PowerShell):

```powershell
git clone https://github.com/gabrielrb/tech-challenge-fase1.git
cd tech-challenge-fase1
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

O `requirements.txt` traz as mesmas versões de bibliotecas do Colab. Ele foi gerado pela seção B13 do notebook da Parte B.

### Problemas comuns

| Problema | O que fazer |
| --- | --- |
| A Parte B mostra `GPU: DESLIGADA` | ligue a GPU (passo 2 da opção 1) e rode de novo |
| O Colab diz que não há GPU disponível | o limite de uso grátis de GPU foi atingido: rode sem GPU (mais lento) ou tente mais tarde |
| O download do Kaggle pede login | crie um token em kaggle.com → **Settings → API**. No Colab, salve-o como segredo `KAGGLE_API_TOKEN`. No Docker, use `docker run --rm -p 127.0.0.1:8888:8888 -e KAGGLE_API_TOKEN=seu_token tech-challenge` |
| A Parte A abre uma janela para escolher um arquivo | o notebook não achou o `data.csv` no GitHub: escolha o `data.csv` baixado deste repositório |
| O PowerShell diz que a execução de scripts está desabilitada | rode `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e repita o `Activate.ps1` (vale só para aquela janela) |

## Estrutura do repositório

```text
tech-challenge-fase1/
├── diagnostico_cancer_mama.ipynb   # Parte A: notebook com as saídas
├── diagnostico_pneumonia.ipynb     # Parte B (EXTRA): notebook com as saídas
├── data.csv                        # base da Parte A
├── figuras/                        # gráficos das Partes A e B, usados no relatório
├── relatorio_tecnico.pdf           # relatório técnico
├── requirements.txt                # bibliotecas e versões do Colab
├── Dockerfile                      # ambiente para rodar fora do Colab
├── .dockerignore                   # arquivos que o Docker ignora
├── .gitignore                      # arquivos que o Git ignora
└── README.md                       # este arquivo
```

## Bases de dados

| Parte | Base | Tamanho | Fonte |
| --- | --- | --- | --- |
| A | Breast Cancer Wisconsin (Diagnostic) | 569 pacientes: 357 benignos e 212 malignos | [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) · [UCI](https://doi.org/10.24432/C5DW2B) |
| B | Chest X-Ray Images (Pneumonia) | 5.856 radiografias: 1.583 NORMAL e 4.273 PNEUMONIA | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) · [Mendeley Data](https://doi.org/10.17632/rscbjbr9sj.3) |

A base da Parte B vem do artigo de Kermany et al. (2018), *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell 172(5). DOI [10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010).

## Tecnologias

| Uso | Ferramentas |
| --- | --- |
| Linguagem e ambiente | Python 3.13, Google Colab, Jupyter Lab |
| Dados e gráficos | pandas, NumPy, Matplotlib, seaborn |
| Machine Learning | scikit-learn, SHAP |
| Imagens e redes neurais | OpenCV, TensorFlow / Keras |
| Base da Parte B | kagglehub |
| Execução fora do Colab | Docker, GitHub Codespaces |

## Limitações

- **Parte A:** o modelo ainda deixou passar 3 de 42 cânceres no teste. A base é pequena e vem de um único centro, nos EUA.
- **Parte B:** a base tem só crianças de 1 a 5 anos, de um único hospital na China. O modelo não vale para adultos nem para outros aparelhos sem um novo teste. No teste, o recall de NORMAL caiu de 0,95 (validação) para 0,41: o modelo ainda dá alarmes falsos demais.
- **As duas partes:** antes de qualquer uso real, os modelos precisam ser testados com dados do próprio hospital. O limiar de decisão deve ser definido com a equipe médica.

## Integrantes

| GitHub |
| --- |
| [@usuario] |
| [@usuario] |

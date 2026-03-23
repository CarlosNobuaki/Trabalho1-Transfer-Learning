# Skin Cancer Classification with Transfer Learning (HAM10000)

## Resumo do projeto
Este projeto treina um classificador de cancer de pele usando Transfer Learning com **EfficientNet-B4** e o dataset **HAM10000**.

O fluxo completo e:
1. Baixar/usar o dataset HAM10000 via Kaggle API.
2. Dividir o dataset em treino, validacao e teste com estratificacao por classe.
3. Treinar em duas etapas (head e fine-tuning).
4. Avaliar com multiplas metricas.
5. Salvar checkpoint e artefatos para inferencia.
6. Classificar novas imagens com o script de deteccao.

## Dataset
- Fonte: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data
- Total de classes: 7

## Tabela de classes usadas no treinamento
| Sigla | Nome da classe | Descricao resumida |
|---|---|---|
| `akiec` | Actinic keratoses / intraepithelial carcinoma | Lesoes pre-malignas e carcinoma intraepitelial |
| `bcc` | Basal cell carcinoma | Carcinoma basocelular |
| `bkl` | Benign keratosis-like lesions | Lesoes benignas tipo queratose |
| `df` | Dermatofibroma | Lesao geralmente benigna de tecido fibroso |
| `mel` | Melanoma | Melanoma maligno |
| `nv` | Melanocytic nevi | Nevos melanociticos (pintas) |
| `vasc` | Vascular lesions | Lesoes vasculares (ex.: angiomas) |

## Como o treinamento funciona
O arquivo `skin-cancer-classify-train.py` executa:

1. **Preparacao e autenticacao Kaggle**
   - Credenciais em `.env`.
   - Aceita `KAGGLE_USERNAME` + `KAGGLE_API_KEY` (mapeando para `KAGGLE_KEY` quando necessario).
   - Baixa o dataset se `HAM10000_metadata.csv` nao existir localmente.

2. **Split de dados (estratificado)**
   - `test_size=0.15` (15% do total)
   - `val_size=0.15` (15% do total)
   - treino recebe os 70% restantes
   - A divisao e estratificada por classe para manter proporcoes.

3. **Transforms de imagem**
   - Treino: resize, flip horizontal aleatorio, rotacao, color jitter e normalizacao ImageNet.
   - Validacao/Teste: resize e normalizacao ImageNet.

4. **Modelo e estrategia de treino**
   - Backbone: EfficientNet-B4 pre-treinada no ImageNet.
   - Cabeca final substituida para 7 classes.
   - Treino em dois estagios:
     - Head (backbone congelada): `epochs_head=5`, `lr_head=1e-3`
     - Fine-tuning (backbone descongelada): `epochs_finetune=10`, `lr_finetune=3e-5`

5. **Otimizacao**
   - Otimizador: `AdamW`
   - Scheduler: `CosineAnnealingLR`
   - Funcao de perda: `CrossEntropyLoss`

6. **Metricas calculadas**
   - Loss
   - Accuracy
   - Balanced Accuracy
   - Precision macro e weighted
   - Recall macro e weighted
   - F1 macro e weighted
   - Log Loss
   - ROC AUC OVR macro (quando aplicavel)
   - Classification Report por classe
   - Matriz de confusao

7. **Artefatos gerados**
   - `artifacts/best_efficientnet_b4_ham10000.pth`
   - `artifacts/class_to_idx.json`
   - `artifacts/training_history.csv`
   - `artifacts/metrics_summary.json`

## Inferencia (classificacao de imagem)
O arquivo `skin-cancer-detect.py` carrega o checkpoint salvo e classifica uma imagem.

- Suporte a device `auto`, `mps`, `cpu`, `cuda`.
- Em MacBook com chip Apple Silicon (M3), `auto` prioriza `mps`.

Exemplo:

```bash
python3 skin-cancer-detect.py --image ./data/HAM10000_images_part_1/ISIC_0024306.jpg --device auto
```

## Execucao recomendada
1. Criar/ativar ambiente virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Configurar `.env` com credenciais Kaggle:

```dotenv
KAGGLE_USERNAME=seu_usuario
KAGGLE_API_KEY=sua_chave
```

4. Treinar:

```bash
python3 skin-cancer-classify-train.py --device auto
```

5. Detectar em imagem nova:

```bash
python3 skin-cancer-detect.py --image caminho_da_imagem.jpg --device auto
```

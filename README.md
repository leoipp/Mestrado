<p align="center">
  <img src="https://portal.ufvjm.edu.br/dicom/central-de-conteudo/identidade-visual/marcas-ufvjm/vertical-sem-assinatura-colorida.png" alt="UFVJM" height="80"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://sglab.com.br/img/logolab/408256cff6d6421eb1b0159939770fb6.png" alt="TreeLab" height="80"/>
</p>

<h1 align="center">Predicao de Volume Florestal com LiDAR</h1>

<p align="center">
  <strong>Dissertacao de Mestrado em Ciencia Florestal</strong><br>
  Universidade Federal dos Vales do Jequitinhonha e Mucuri - UFVJM
</p>

<p align="center">
  <strong>Mestrando:</strong> Leonardo Ippolito Rodrigues<br>
  <strong>Orientador:</strong> Prof. Dr. Eric Bastos Gorgens<br>
  <strong>Grupo de Pesquisa:</strong> <a href="https://sglab.com.br/site/laboratorio-de-ciencia-de-dados-florestais/177/">TreeLab - Forest Data Science</a>
</p>

---

## Introducao

O manejo florestal de precisao demanda estimativas acuradas de volume de madeira para otimizar a tomada de decisao em plantios comerciais. Tradicionalmente, essas estimativas sao obtidas atraves de inventarios florestais convencionais, que envolvem medicoes em campo de diametro e altura de arvores amostrais - um processo custoso, demorado e limitado espacialmente.

O LiDAR (Light Detection and Ranging) aerotransportado representa uma revolucao na caracterizacao estrutural de florestas, permitindo a aquisicao de milhoes de pontos tridimensionais que descrevem a arquitetura do dossel com precisao centimetrica. A partir dessas nuvens de pontos, e possivel extrair metricas relacionadas a altura, densidade e distribuicao vertical da vegetacao, que apresentam forte correlacao com atributos biofisicos como biomassa e volume.

Este projeto desenvolve um pipeline completo para **predicao de volume florestal (VTCC - Volume Total Com Casca)** em plantios de *Eucalyptus* spp., integrando:

- **Processamento de dados LiDAR** com LAStools
- **Extracao de metricas** de altura do dossel (percentis, maximos, curtose)
- **Selecao de variaveis** via Recursive Feature Elimination (RFE)
- **Modelagem preditiva** com Random Forest e Redes Neurais (MLP)
- **Espacializacao** das estimativas em formato raster

O diferencial desta abordagem reside na combinacao de variaveis derivadas exclusivamente do LiDAR com informacoes cadastrais (idade, rotacao, regional), permitindo a geracao de mapas continuos de volume sem a necessidade de interpolacao de dados pontuais.

---

## Estrutura do Projeto

```
Mestrado/
│
├── Auxiliary/                        # Scripts auxiliares
│   ├── LidarPreProcessor.py          # Pipeline de processamento LiDAR (LAStools)
│   ├── CalculoCubMean.py             # Calculo da media cubica de elevacao
│   ├── AsciiToTiff.py                # Conversao ASCII para GeoTIFF
│   ├── MergeTiff.py                  # Mosaico de rasters
│   ├── ShapeToRaster.py              # Rasterizacao de shapefiles
│   └── PredictVolume.py              # Aplicacao do modelo em rasters
│
├── Forecast/
│   ├── Predictive/                   # Modelagem preditiva
│   │   ├── 01_DataConsistency.py         # Limpeza e consistencia dos dados
│   │   ├── 02_VariablesCorrelation.py    # Analise de correlacao e RFE
│   │   ├── 03_FeatureSelection.py        # Selecao de variaveis
│   │   ├── 04_RandomForestTrain.py       # Treinamento Random Forest
│   │   ├── 05_NeuralNetworkTrain.py      # Treinamento MLP (PyTorch)
│   │   ├── ValidateRFR.py                # Validacao do modelo
│   │   ├── Models/                       # Modelos treinados (.pkl, .pt)
│   │   └── Results/                      # Metricas e graficos
│   │
│   └── Projection/                   # Projecao temporal [EM CONSTRUCAO]
│
└── Data/                             # Dados (nao versionados)
    └── DataFrames/
```

---

## Pipeline de Processamento

### 1. Pre-processamento LiDAR

O script `LidarPreProcessor.py` implementa um pipeline completo de processamento de nuvens de pontos utilizando ferramentas do [LAStools](https://rapidlasso.com/lastools/):

| Etapa | Processo | Ferramenta | Descricao |
|:-----:|----------|------------|-----------|
| 1 | Catalogo | lasinfo | Estatisticas da nuvem de pontos |
| 2 | Tiling | lastile | Divisao em tiles para processamento paralelo |
| 3 | Denoising | lasnoise | Remocao de ruidos e pontos isolados |
| 4 | Ground | lasground_new | Classificacao de pontos de solo |
| 5 | DTM | las2dem | Modelo Digital de Terreno |
| 6 | Thinning | lasthin | Reducao de densidade |
| 7 | Normalizacao | lasground_new | Altura acima do solo (z - DTM) |
| 8 | CHM | lasgrid | Modelo de Altura de Copa |
| 9 | DSM | lasgrid | Modelo Digital de Superficie |
| 10 | Metricas | lascanopy | Percentis, maximo, curtose |

```bash
python LidarPreProcessor.py -i dados/*.las -o resultados/ --metrics-res 17 --percentiles 60 90
```

### 2. Modelagem Preditiva

#### 2.1 Consistencia dos Dados
Remocao de outliers, tratamento de valores ausentes e validacao de intervalos.

#### 2.2 Analise de Correlacao
Matriz de correlacao de Pearson e ranking de variaveis via RFE (Recursive Feature Elimination).

#### 2.3 Selecao de Variaveis
Avaliacao de combinacoes de variaveis com validacao cruzada K-Fold.

#### 2.4 Treinamento de Modelos

| Modelo | Script | Otimizacao | Saidas |
|--------|--------|------------|--------|
| **Random Forest** | `04_RandomForestTrain.py` | RandomizedSearchCV | `RF_Regressor.pkl` |
| **MLP** | `05_NeuralNetworkTrain.py` | Optuna | `MLP_Regressor.pt` |

Ambos os modelos geram:
- Arquivo do modelo treinado
- `*_Training_Metrics.xlsx` - Metricas de validacao cruzada
- `*_Diagnostics.png` - Graficos de diagnostico
- `*_Feature_Importance.png` - Importancia das variaveis

### 3. Aplicacao Espacial

O script `PredictVolume.py` aplica o modelo treinado em rasters para gerar mapas continuos de volume:

```bash
python PredictVolume.py --model RF_Regressor.pkl --output resultados/ \
    --p90 p90.tif --p60 p60.tif --max max.tif \
    --rotacao rotacao.tif --regional regional.tif --idade idade.tif
```

**Saidas:**
- `*_volume_estimado.tif` - Predicao de VTCC (m³/ha)
- `*_volume_incerteza.tif` - Incerteza da predicao (desvio padrao entre arvores do RF)

---

## Variaveis do Modelo

| Variavel | Tipo | Descricao | Fonte |
|----------|------|-----------|-------|
| **Elev P90** | LiDAR | Percentil 90 de altura normalizada | lascanopy |
| **Elev P60** | LiDAR | Percentil 60 de altura normalizada | lascanopy |
| **Elev maximum** | LiDAR | Altura maxima | lascanopy |
| **ROTACAO** | Cadastral | Numero da rotacao do plantio | Shapefile |
| **REGIONAL** | Cadastral | Codigo da unidade regional | Shapefile |
| **Idade (meses)** | Cadastral | Idade do plantio em meses | Shapefile |

---

## Metricas de Avaliacao

| Metrica | Descricao |
|---------|-----------|
| **R²** | Coeficiente de determinacao |
| **RMSE** | Raiz do erro quadratico medio (m³/ha) |
| **MAE** | Erro absoluto medio (m³/ha) |
| **Bias** | Vies medio (m³/ha) |
| **RMSE%** | RMSE relativo a media |
| **MAE%** | MAE relativo a media |

---

## Dependencias

```bash
# Analise de dados
pip install numpy pandas openpyxl

# Machine Learning
pip install scikit-learn

# Visualizacao
pip install matplotlib seaborn

# Geoespacial
pip install rasterio shapely fiona

# Deep Learning (para MLP)
pip install torch optuna

# Processamento LiDAR
# LAStools: https://rapidlasso.com/lastools/
```

---

## Requisitos

- Python 3.8+
- LAStools (processamento de nuvens de pontos)
- 8 GB RAM (recomendado 16 GB para grandes areas)

---

## Uso Rapido

```python
# Treinar Random Forest
from Forecast.Predictive.04_RandomForestTrain import train_random_forest
results = train_random_forest()
print(f"R² = {results['metrics']['R2']:.4f}")

# Treinar MLP
from Forecast.Predictive.05_NeuralNetworkTrain import train_mlp
results = train_mlp()

# Aplicar modelo em rasters
from Auxiliary.PredictVolume import predict_volume
predict_volume(
    raster_paths={
        'Elev P90': 'metricas/p90.tif',
        'Elev P60': 'metricas/p60.tif',
        'Elev maximum': 'metricas/max.tif',
        'ROTACAO': 'cadastro/rotacao.tif',
        'REGIONAL': 'cadastro/regional.tif',
        'Idade (meses)': 'cadastro/idade.tif'
    },
    model='Models/RF_Regressor.pkl',
    output_dir='resultados/'
)
```

---

## Projection

> **EM CONSTRUCAO**
>
> Modulo para projecao temporal do crescimento florestal utilizando series temporais de dados LiDAR.

---

## Referencias


---

## Licenca

Este projeto e parte de uma dissertacao de mestrado e esta disponivel para fins academicos e de pesquisa.

---

<p align="center">
  <strong>TreeLab - Forest Data Science</strong><br>
  Universidade Federal dos Vales do Jequitinhonha e Mucuri<br>
  Diamantina, MG - Brasil
</p>


# Análise de Técnicas de Redução de Amostragem em Sistemas de Orquestração Inteligentes

Este repositório contém os códigos e scripts da pesquisa desenvolvida por Davi Felipe Ramos de Oliveira Vilarinho na Universidade Federal de Uberlândia, intitulada:

    "Análise de Técnicas de Redução de Amostragem em Sistemas de Orquestração Inteligentes"

## 🌐 **Contexto**

O objetivo central da pesquisa é compreender como a performance de modelos de Machine Learning se deteriora quando há uma diferença entre os dados utilizados no treinamento e os dados reais recebidos em produção. Isso é especialmente importante em contextos onde a coleta de dados do lado do cliente é limitada por questões de privacidade, desempenho ou disponibilidade.

Com base em estudos anteriores conduzidos pelo professor Pasquini, que demonstraram a viabilidade de prever QoS (Qualidade de Serviço) em aplicações de Streaming e Bancos de Dados usando árvores de regressão e Random Forest, esta pesquisa investiga como a periodicidade dos dados (por segundo, minuto etc.) afeta essa capacidade preditiva.

### Resumo

Este trabalho investiga o impacto da sobrecarga causada pela grande quantidade de métricas coletadas e transmitidas em sistemas de monitoramento, especialmente em ambientes de orquestração com aprendizado de máquina. São propostas e avaliadas três estratégias para reduzir a frequência de coleta das amostras: (i) eliminação de amostras intermediárias, (ii) agregação em janelas com funções estatísticas e (iii) representação das janelas por meio de oito estatísticas descritivas para cada métrica.

A eliminação de amostras intermediárias evidenciou a importância do equilíbrio entre o número de características e o volume de amostras. Os melhores resultados foram obtidos com janelas de 128 e 256 segundos apenas quando aplicada a seleção de características — sem ela, o desempenho caiu significativamente, limitando-se a janelas de 16 segundos. Já a agregação com funções estatísticas permitiu janelas de até 64 segundos, mas apresentou distorções na distribuição das amostras em janelas maiores. Por fim, a descrição das métricas com oito estatísticas permitiu janelas de até 128 segundos, desde que acompanhadas por seleção de características para lidar com a alta dimensionalidade; sem essa técnica, o limite eficiente foi de 32 segundos.

Conclui-se que a redução da frequência de coleta é viável, desde que se mantenha um equilíbrio entre dimensionalidade e quantidade de amostras, sendo a seleção de características essencial em cenários com maior densidade de atributos.

## 🛠️ **Organização e Execução dos Experimentos**

Os experimentos foram organizados em duas grandes etapas:

1. **Geração dos Dados**

    Utilizamos scripts que simulam o comportamento de agregadores para gerar os datasets pré-processados com diferentes granularidades temporais e abordagens:

    - `agg_function_periodic_dataset_generator_parallel.py`
    - `distrib_periodic_dataset_generator_parallel.py`

    Esses scripts devem ser executados primeiro, e são responsáveis por gerar os dados com diferentes resoluções temporais e características (por função ou por distribuição).

2. **Execução dos Experimentos**

    Com os dados gerados, os seguintes arquivos executam os modelos e validam os comportamentos observados:

    **Modelos Naive**:

    - `naive_periodic_experiment.py`
    - `naive_periodic_experiment_k_fold.py`
    - `naive_periodic_experiment_tptt.py`

    **Modelos baseados em Distribuição**:

    - `distrib_function_periodic_experiment_y_original.py`
    - `distrib_function_periodic_experiment_y_original_k_fold.py`
    - `distrib_tptt_function_periodic_experiment_y_original.py`
    - `distrib_tptt_function_periodic_experiment_y_original_fs_multiplied.py`

    **Modelos baseados em Função Agregadora**:

    - `agg_function_periodic_experiment_y_original.py`
    - `agg_function_periodic_experiment_y_original_kfold.py`
    - `agg_tptt_function_periodic_experiment_y_original.py`

    Esses scripts permitem comparar diferentes granularidades e abordagens de modelagem sobre o impacto da periodicidade nos dados. Os experimentos incluem também validações por K-Fold e testes com TPTT (Time-Preserving Train-Test).

🔍 **Experimento Especial: `experiment_4_7200samples.py`**

Este experimento foi uma tentativa de trabalhar com 7.200 amostras, avaliando o comportamento do modelo quando exposto a um conjunto de dados mais denso, anterior à reformulação geral do fluxo de experimentação. Serve como referência comparativa.

## 📊 **Resultados e Análises**

As análises, embora extensivamente feitas via Jupyter Notebooks, foram condensadas e replicadas em scripts standalone, de forma que os principais resultados podem ser reproduzidos diretamente a partir dos arquivos `.py`.

## 📁 **Estrutura do Repositório**

- `*_dataset_generator_*.py` → Scripts para geração de dados com diferentes abordagens.
- `*_experiment_*.py` → Execução dos experimentos com diferentes técnicas.
- demais arquivos: em geral histórico.

## ✅ **Requisitos**

- Python 3.10+
- Scikit-learn
- Pandas / NumPy
- Matplotlib / Seaborn (para análises gráficas opcionais)
- (Opcional) JupyterLab para visualização e análise interativa

## 📚 **Citação**

Considere me citar 😄

📚 **ABNT (Associação Brasileira de Normas Técnicas)**

> VILARINHO, Davi Felipe Ramos de Oliveira. Análise de Técnicas de Redução de Amostragem em Sistemas de Orquestração Inteligentes. Uberlândia: Universidade Federal de Uberlândia, 2025.

🔤 **APA (American Psychological Association)**

> Vilarinho, D. F. R. de O. (2025). Análise de técnicas de redução de amostragem em sistemas de orquestração inteligentes. Universidade Federal de Uberlândia.

🔢 **BibTeX (para usar em LaTeX)**


```
@bachelorsthesis{vilarinho2025amostragem,
  author       = {Davi Felipe Ramos de Oliveira Vilarinho},
  title        = {Análise de Técnicas de Redução de Amostragem em Sistemas de Orquestração Inteligentes},
  school       = {Universidade Federal de Uberlândia},
  year         = {2025},
  type         = {Trabalho de Conclusão de Curso (Graduação)}
}
```

📘 **Chicago**

> Vilarinho, Davi Felipe Ramos de Oliveira. Análise de Técnicas de Redução de Amostragem em Sistemas de Orquestração Inteligentes. Uberlândia: Universidade Federal de Uberlândia, 2025.


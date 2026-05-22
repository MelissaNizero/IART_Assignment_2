# Employee Burnout Risk Prediction

Projeto de Machine Learning para prever o risco de burnout de colaboradores em tres classes: `Low`, `Medium` e `High`. O foco do trabalho e o modelo preditivo e a sua avaliacao empirica, sem desenvolvimento de uma aplicação ou interface.

## Objetivo

O objetivo e detetar sinais precoces de exaustao profissional atraves de dados de atividade laboral, bem-estar, feedback em inqueritos "Pulse" e perfil do colaborador. Esta previsao permite que a organizacao atue preventivamente antes de ocorrerem baixas medicas prolongadas, pedidos de demissao ou perda acentuada de produtividade.

## Dataset

Foi gerado um dataset sintetico com 1000 colaboradores. A geracao inclui correlacoes estatisticas plausiveis:

- mais horas extraordinarias aumentam o risco;
- muitos dias desde as ultimas ferias aumentam o risco;
- maior volume de reunioes aumenta o risco;
- stress auto-reportado alto aumenta o risco;
- motivacao baixa aumenta o risco;
- faltas/baixas recentes contribuem para maior risco.

Distribuicao gerada da variavel-alvo:

| Burnout Risk | Registos |
| --- | ---: |
| Low | 370 |
| Medium | 370 |
| High | 260 |

## Variaveis

| Categoria | Features |
| --- | --- |
| Activity | `total_weekly_hours`, `overtime_hours`, `daily_meeting_volume` |
| Well-being | `days_since_last_vacation`, `sick_leave_days_6m` |
| Feedback | `self_reported_stress`, `motivation_level` |
| Profile | `department`, `role`, `seniority_years` |
| Target | `burnout_risk` |

## Metodologia

- Linguagem: Python
- Bibliotecas: pandas, scikit-learn, matplotlib e seaborn
- Suporte adicional: python-pptx para gerar uma copia preenchida da apresentacao
- Web app: servidor HTTP simples em Python para demonstrar a previsao do modelo
- Divisao dos dados: 80% treino / 20% teste
- Estrategia: Hold-out com `random_state=42` e estratificacao pela variavel-alvo
- Modelos comparados:
  - Logistic Regression
  - SVM
  - Random Forest
- Metricas:
  - Precision
  - Recall
  - F1-Score

## Resultados

Resultados no conjunto de teste com 200 registos:

| Modelo | Precision Macro | Recall Macro | F1 Macro | Accuracy aproximada |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.851 | 0.858 | 0.854 | 0.855 |
| SVM | 0.848 | 0.848 | 0.848 | 0.850 |
| Random Forest | 0.813 | 0.818 | 0.815 | 0.815 |

O melhor modelo foi a **Logistic Regression**, com F1 macro de aproximadamente 0.854. O resultado indica que, neste dataset sintetico, a fronteira entre classes e suficientemente estruturada para um modelo linear capturar bem os padroes principais. A classe `Medium` e a mais dificil, porque representa a zona de transicao entre baixo e alto risco.

## Como executar

Instalar dependencias:

```bash
python -m pip install -r requirements.txt
```

Gerar o dataset:

```bash
python main.py generate
```

Treinar e avaliar os modelos:

```bash
python main.py train
```

Atualizar a apresentacao a partir do template em `Downloads`:

```bash
python main.py presentation
```

Executar a web app:

```bash
python main.py app
```

Depois abrir no browser:

```text
http://127.0.0.1:8000
```

Na interface e possivel inserir dados do trabalhador, incluindo nome, idade,
departamento, cargo, carga horaria, horas extra, ferias, baixas, stress e
motivacao. Depois da previsao, a pagina gera um relatorio com o risco previsto,
probabilidades do modelo e sugestoes para os niveis Low, Medium e High. O botao
`Print report` abre a impressao do relatorio.

## Estrutura

```text
.
|-- data/
|   `-- employee_burnout_synthetic.csv
|-- results/
|   |-- classification_reports.txt
|   |-- dataset_summary.csv
|   |-- model_comparison.csv
|   `-- figures/
|       |-- logistic_regression_confusion_matrix.png
|       |-- model_comparison.png
|       |-- random_forest_confusion_matrix.png
|       `-- svm_confusion_matrix.png
|-- src/
|   |-- __init__.py
|   |-- generate_dataset.py
|   |-- train_evaluate.py
|   |-- update_presentation.py
|   `-- web_app.py
|-- main.py
|-- docs/
|   `-- slide8_results.md
|-- presentation/
|   `-- Employee_Burnout_Risk_Prediction.pptx
|-- requirements.txt
`-- README.md
```

## Conclusao

O estudo mostra que e possivel construir um classificador multi-classe para estimar risco de burnout a partir de indicadores operacionais e de bem-estar. A Logistic Regression teve o melhor equilibrio geral entre precision e recall, seguida de perto pelo SVM. Em contexto real, o modelo deveria ser usado apenas como ferramenta de apoio a decisao e acompanhado por regras claras de privacidade, transparencia e nao discriminacao.

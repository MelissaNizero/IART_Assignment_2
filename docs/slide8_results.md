# Slide 8 - Empirical Study: Results

## Tabela para o slide

| Algorithm | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.851 | 0.858 | 0.854 |
| SVM | 0.848 | 0.848 | 0.848 |
| Random Forest | 0.813 | 0.818 | 0.815 |

## Texto curto para apresentar

Foram comparados tres classificadores usando uma divisao Hold-out 80/20. O melhor
resultado foi obtido pela Logistic Regression, com F1 macro de 0.854. O SVM teve
um desempenho muito proximo, enquanto o Random Forest ficou ligeiramente abaixo.

A classe mais facil de identificar foi `Low`, porque os padroes de baixo risco
sao mais claros. A classe `Medium` teve mais confusoes, o que e esperado por ser
uma zona intermedia entre sinais saudaveis e sinais criticos de burnout.

## Conclusao para o slide

O modelo e promissor para apoio a decisao preventiva, mas deve ser usado com
cuidado etico: a previsao deve orientar apoio ao colaborador, nao punicao ou
vigilancia individual excessiva.

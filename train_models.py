import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
import joblib

np.random.seed(42)
n_samples = 1200

horas_extra = np.random.randint(0, 25, n_samples)
dias_ferias = np.random.randint(0, 22, n_samples)
nivel_stress = np.random.randint(1, 6, n_samples)
meses_continuos = np.random.randint(1, 12, n_samples)

score = (horas_extra * 0.4) - (dias_ferias * 0.3) + (nivel_stress * 0.7) + (meses_continuos * 0.3)
burnout = (score > np.median(score)).astype(int)

ruido = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
burnout[ruido] = 1 - burnout[ruido]

df = pd.DataFrame({
    'Horas_Extra': horas_extra,
    'Dias_Ferias': dias_ferias,
    'Nivel_Stress': nivel_stress,
    'Meses_Continuos': meses_continuos,
    'Burnout': burnout
})

df.to_csv("burnout_data.csv", index=False)

X = df.drop('Burnout', axis=1)
y = df['Burnout']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos = {
    "Logistic_Regression": LogisticRegression(),
    "Decision_Tree": DecisionTreeClassifier(max_depth=4),
    "Random_Forest": RandomForestClassifier(n_estimators=100)
}

melhor_modelo = None
melhor_score = 0
nome_vencedor = ""

print("--- Avaliação de Desempenho dos Algoritmos ---")
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    preds = modelo.predict(X_test)

    score_aval = f1_score(y_test, preds)
    rec = recall_score(y_test, preds)
    print(f"{nome} -> F1-Score: {score_aval:.2f} | Recall: {rec:.2f}")
    
    if score_aval > melhor_score:
        melhor_score = score_aval
        melhor_modelo = modelo
        nome_vencedor = nome

print(f"\n🏆 O vencedor foi: {nome_vencedor} com F1 de {melhor_score:.2f}")
joblib.dump(melhor_modelo, "melhor_modelo_burnout.pkl")
print("Modelo guardado com sucesso em 'melhor_modelo_burnout.pkl'!")
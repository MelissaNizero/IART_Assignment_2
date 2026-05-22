import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Burnout Detective", layout="centered")

st.title("🛡️ BurnoutShield AI")
st.markdown("### Prevenção e Alerta Precoce de Riscos Psicossociais")

try:
    modelo = joblib.load("melhor_modelo_burnout.pkl")
except:
    st.error("Por favor, corre primeiro o script 'train_models.py' para gerar o modelo!")
    st.stop()

st.write("Insira os dados métricos do colaborador para calcular o risco atual:")

col1, col2 = st.columns(2)
with col1:
    horas = st.slider("Horas Extra / Semana", 0, 25, 5)
    ferias = st.slider("Dias de Férias Gozados (Ano)", 0, 22, 12)
with col2:
    stress = st.slider("Nível de Stress Autorreportado", 1, 5, 3)
    meses = st.slider("Meses de Trabalho Contínuo", 1, 12, 4)

input_dados = pd.DataFrame([[horas, ferias, stress, meses]], 
                           columns=['Horas_Extra', 'Dias_Ferias', 'Nivel_Stress', 'Meses_Continuos'])

if st.button("Analisar Risco com IA"):
    previsao = modelo.predict(input_dados)[0]
    probabilidade = modelo.predict_proba(input_dados)[0][1] * 100
    
    st.subheader("Veredito Técnico:")
    if previsao == 1:
        st.error(f"🚨 **Risco Crítico de Burnout Ativo ({probabilidade:.1f}%)**")
        st.warning("⚠️ **Recomendações urgentes de Gestão:** Reduzir carga horária imediatamente e agendar pausas compensatórias.")
    else:
        st.success(f"✅ **Indicadores Saudáveis ({probabilidade:.1f}% de probabilidade de risco)**")
        st.info("O colaborador encontra-se dentro de parâmetros de equilíbrio psicossocial estáveis.")
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# =======================
# FUNÇÕES AUXILIARES
# =======================
def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8').lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def extrair_variavel(padrao, texto, tipo=float, valor_padrao=None):
    match = re.search(padrao, texto)
    if match:
        try:
            return tipo(match.group(1))
        except:
            return valor_padrao
    return valor_padrao

def treinar_modelos(df, le_mob, le_app):
    # Preparação dos dados para eutanásia
    X_eutanasia = df[features_eutanasia].copy()
    y_eutanasia = df['Eutanasia'].copy()
    # Convertendo para float/int e tratando NaNs
    X_eutanasia = X_eutanasia.astype(float).fillna(X_eutanasia.mean())
    y_eutanasia = y_eutanasia.astype(int).fillna(y_eutanasia.mode()[0])
    X_train, X_test, y_train, y_test = train_test_split(
        X_eutanasia, y_eutanasia, test_size=0.2, random_state=42, stratify=y_eutanasia
    )
    # Sem SMOTE, treina direto
    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_eutanasia.fit(X_train, y_train)

    # Preparação dos dados para alta
    X_alta = df[features].copy()
    y_alta = df['Alta'].copy()
    X_alta = X_alta.astype(float).fillna(X_alta.mean())
    y_alta = y_alta.astype(int).fillna(y_alta.mode()[0])
    X_alta_train, _, y_alta_train, _ = train_test_split(
        X_alta, y_alta, test_size=0.2, random_state=42, stratify=y_alta
    )
    modelo_alta = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_alta.fit(X_alta_train, y_alta_train)

    # Modelo para internar
    modelo_internar = RandomForestClassifier(random_state=42)
    modelo_internar.fit(df[features].astype(float), df['Internar'].astype(int))

    # Modelo para dias internado (apenas pacientes internados)
    df_internados = df[df['Internar'] == 1]
    modelo_dias = RandomForestClassifier(random_state=42)
    modelo_dias.fit(df_internados[features].astype(float), df_internados['Dias Internado'].astype(int))

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias

# =======================
# CARREGAMENTO DE DADOS
# =======================
df = pd.read_csv("Casos_Cl_nicos_Simulados.csv")
df_doencas = pd.read_csv("doencas_caninas_eutanasia_expandidas.csv")

palavras_chave_eutanasia = [
    unicodedata.normalize('NFKD', d).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    for d in df_doencas['Doença'].dropna().unique()
]

le_mob = LabelEncoder()
le_app = LabelEncoder()
df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())

df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
    lambda d: int(any(p in unicodedata.normalize('NFKD', d).encode('ASCII', 'ignore').decode('utf-8').lower()
                      for p in palavras_chave_eutanasia))
)

features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
features_eutanasia = features + ['tem_doenca_letal']

modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = treinar_modelos(df, le_mob, le_app)

# =======================
# INTERFACE STREAMLIT
# =======================
st.title("💉 Avaliação Clínica Canina")

if "mostrar_resultado" not in st.session_state:
    st.session_state.mostrar_resultado = False
if "resultado" not in st.session_state:
    st.session_state.resultado = None

anamnese = st.text_area("📝 Digite a anamnese do paciente:")

col1, col2 = st.columns(2)
with col1:
    if st.button("🔍 Analisar"):
        st.session_state.resultado = prever(anamnese)
        st.session_state.mostrar_resultado = True

with col2:
    if st.button("🚪 Fechar Programa"):
        st.write("✅ Programa encerrado. Pode fechar a aba do navegador.")
        st.stop()

if st.session_state.mostrar_resultado and st.session_state.resultado:
    st.subheader("📋 Resultado da Avaliação:")
    for chave, valor in st.session_state.resultado.items():
        st.write(f"**{chave}**: {valor}")

    if st.button("🔁 Analisar outra anamnese"):
        st.session_state.mostrar_resultado = False
        st.session_state.resultado = None
        st.experimental_rerun()



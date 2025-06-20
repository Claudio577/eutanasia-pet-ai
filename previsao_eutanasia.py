import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ========= FUNÇÕES AUXILIARES =========
def normalizar_texto(texto):
    texto = unicodedata.normalize('NFKD', str(texto)).encode('ASCII', 'ignore').decode('utf-8').lower()
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
    features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
    features_eutanasia = features + ['tem_doenca_letal']

    # Modelo Eutanásia
    X_eutanasia = df[features_eutanasia]
    y_eutanasia = df['Eutanasia']
    X_train, _, y_train, _ = train_test_split(X_eutanasia, y_eutanasia, test_size=0.2, random_state=42, stratify=y_eutanasia)
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_eutanasia.fit(X_train_res, y_train_res)

    # Modelo Alta
    X_alta = df[features]
    y_alta = df['Alta']
    X_alta_train, _, y_alta_train, _ = train_test_split(X_alta, y_alta, test_size=0.2, random_state=42, stratify=y_alta)
    X_alta_res, y_alta_res = SMOTE(random_state=42).fit_resample(X_alta_train, y_alta_train)
    modelo_alta = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_alta.fit(X_alta_res, y_alta_res)

    # Modelo Internar e Dias Internado
    modelo_internar = RandomForestClassifier(random_state=42).fit(df[features], df['Internar'])
    modelo_dias = RandomForestRegressor(random_state=42).fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, features, features_eutanasia

# ========= FUNÇÃO DE PREVISÃO =========
def prever(texto, idade, peso, gravidade, dor, mobilidade, apetite, temperatura):
    texto_norm = normalizar_texto(texto)

    # Detecção de doenças
    doencas_detectadas = []
    for d in palavras_chave_eutanasia:
        if d in texto_norm:
            doencas_detectadas.append(d)
        else:
            partes = d.split()
            if all(p in texto_norm for p in partes if len(p) > 3):
                doencas_detectadas.append(d)

    st.write("✅ Doenças detectadas:", doencas_detectadas)
    st.write("🔍 Texto normalizado:", texto_norm)
    st.write("🚩 Quantidade de doenças letais detectadas:", len(doencas_detectadas))

    tem_doenca_letal = 1 if len(doencas_detectadas) > 0 else 0

    dados = [[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]]
    dados_df = pd.DataFrame(dados, columns=features_eutanasia)

    alta = modelo_alta.predict(dados_df[features])[0]
    internar = int(modelo_internar.predict(dados_df[features])[0])
    dias = int(round(modelo_dias.predict(dados_df[features])[0])) if internar == 1 else 0

    eutanasia_chance_model = round(modelo_eutanasia.predict_proba(dados_df)[0][1] * 100, 1)
    st.write(f"🔢 Chance de eutanásia pelo modelo antes do ajuste: {eutanasia_chance_model}%")

    # Ajuste final da chance de eutanásia
    if len(doencas_detectadas) >= 1:
        eutanasia_chance = 95.0
        st.write("⚠️ Forçando chance de eutanásia para 95% por doença letal detectada")
    else:
        if (
            dor >= 7 or 
            apetite == le_app.transform(["nenhum"])[0] or 
            mobilidade == le_mob.transform(["sem andar"])[0] or 
            temperatura > 40 or 
            gravidade == 10
        ):
            eutanasia_chance = max(eutanasia_chance_model, 50.0)
        else:
            eutanasia_chance = eutanasia_chance_model

    st.write(f"✅ Chance final de eutanásia: {eutanasia_chance}%")

    return {
        "Alta": "Sim" if alta == 1 else "Não",
        "Internar": "Sim" if internar == 1 else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": eutanasia_chance,
        "Doenças Detectadas": doencas_detectadas if doencas_detectadas else ["Nenhuma grave"]
    }

# ========= CARREGAMENTO DE DADOS =========
df = pd.read_csv("/mnt/data/Casos_Cl_nicos_Simulados.csv")
df_doencas = pd.read_csv("/mnt/data/doencas_caninas_eutanasia_expandidas.csv")

# Lista de doenças normalizadas
palavras_chave_eutanasia = [
    normalizar_texto(d) for d in df_doencas['Doença'].dropna().unique()
]

# Preparação dos dados
le_mob = LabelEncoder()
le_app = LabelEncoder()
df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())

df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
    lambda d: int(any(p in normalizar_texto(d) for p in palavras_chave_eutanasia))
)

modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias, features, features_eutanasia = treinar_modelos(df, le_mob, le_app)

# ========= INTERFACE STREAMLIT =========
st.title("💉 Avaliação Clínica Canina")

anamnese = st.text_area("Digite a anamnese do paciente:")

# Captura das variáveis numéricas e categóricas para o modelo
idade = st.number_input("Idade (anos)", min_value=0, max_value=30, value=5)
peso = st.number_input("Peso (kg)", min_value=0.1, max_value=100.0, value=10.0)
gravidade = st.slider("Gravidade (0 a 10)", 0, 10, 5)
dor = st.slider("Dor (0 a 10)", 0, 10, 3)

# Mobilidade e Apetite como selectbox com valores usados no label encoder
opcoes_mobilidade = sorted(df['Mobilidade'].astype(str).unique())
opcoes_apetite = sorted(df['Apetite'].astype(str).unique())

# Convertendo as opções para labels legíveis para o usuário
# Mas para simplificar, vamos usar os valores codificados, pois o label encoder usa números
mobilidade = st.selectbox("Mobilidade", options=opcoes_mobilidade, index=0)
apetite = st.selectbox("Apetite", options=opcoes_apetite, index=0)

# Como a label encoder transformou as strings para números, precisamos transformar de volta as strings dos options
# Porém para usar no modelo, precisamos do valor codificado:
mobilidade_cod = int(mobilidade)
apetite_cod = int(apetite)

temperatura = st.number_input("Temperatura (°C)", min_value=30.0, max_value=45.0, value=38.5)

if st.button("Analisar"):
    if anamnese.strip() == "":
        st.warning("Por favor, digite a anamnese para análise.")
    else:
        resultado = prever(
            anamnese,
            idade=idade,
            peso=peso,
            gravidade=gravidade,
            dor=dor,
            mobilidade=mobilidade_cod,
            apetite=apetite_cod,
            temperatura=temperatura
        )
        st.subheader("📋 Resultado da Avaliação:")
        for chave, valor in resultado.items():
            if isinstance(valor, list):
                valor = ", ".join(valor)
            st.write(f"**{chave}**: {valor}")




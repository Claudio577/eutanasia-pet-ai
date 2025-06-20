import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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

def heuristicas_para_valores_reais(nova_linha, le_mob, le_app):
    dor = nova_linha['Dor']
    temp = nova_linha['Temperatura']
    mobilidade = nova_linha['Mobilidade']
    apetite = nova_linha['Apetite']
    tem_doenca_letal = nova_linha['tem_doenca_letal']

    alta = 1 if (
        tem_doenca_letal == 0 and
        dor <= 4 and
        37.5 <= temp <= 39.0 and
        mobilidade == le_mob.transform(['normal'])[0] and
        apetite == le_app.transform(['normal'])[0]
    ) else 0

    internar = 1 if (
        dor >= 7 or
        temp > 39.5 or temp < 37 or
        mobilidade in [le_mob.transform(['sem andar'])[0], le_mob.transform(['limitada'])[0]] or
        apetite in [le_app.transform(['nenhum'])[0], le_app.transform(['baixo'])[0]]
    ) else 0

    if internar == 0:
        dias = 0
    else:
        dias = 7 if tem_doenca_letal == 1 or dor >= 8 or temp > 40 else 3

    eutanasia = 1 if (
        tem_doenca_letal == 1 or
        dor >= 9 or
        apetite == le_app.transform(['nenhum'])[0] or
        mobilidade == le_mob.transform(['sem andar'])[0] or
        temp > 40
    ) else 0

    return alta, internar, dias, eutanasia

def treinar_modelos(df, le_mob, le_app):
    X_eutanasia = df[features_eutanasia]
    y_eutanasia = df['Eutanasia']
    X_train, X_test, y_train, y_test = train_test_split(X_eutanasia, y_eutanasia, test_size=0.2, random_state=42, stratify=y_eutanasia)
    X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
    modelo_eutanasia = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_eutanasia.fit(X_train_res, y_train_res)

    X_alta = df[features]
    y_alta = df['Alta']
    X_alta_train, _, y_alta_train, _ = train_test_split(X_alta, y_alta, test_size=0.2, random_state=42, stratify=y_alta)
    X_alta_res, y_alta_res = SMOTE(random_state=42).fit_resample(X_alta_train, y_alta_train)
    modelo_alta = RandomForestClassifier(class_weight='balanced', random_state=42)
    modelo_alta.fit(X_alta_res, y_alta_res)

    modelo_internar = RandomForestClassifier(random_state=42).fit(df[features], df['Internar'])
    modelo_dias = RandomForestRegressor(random_state=42).fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias

def prever(texto):
    texto_norm = normalizar_texto(texto)

    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", texto_norm, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", texto_norm, float, 10.0)
    temperatura = extrair_variavel(r"(\d{2}(?:\.\d+)?)\s*(?:graus|c|celsius|ºc)", texto_norm, float, 38.5)
    gravidade = 10 if "vermelho" in texto_norm else 5

    if "dor intensa" in texto_norm:
        dor = 10
    elif "dor moderada" in texto_norm:
        dor = 5
    elif "sem dor" in texto_norm:
        dor = 0
    else:
        dor = 4

    if "nenhum apetite" in texto_norm:
        apetite = le_app.transform(["nenhum"])[0]
    elif "baixo apetite" in texto_norm or "apetite baixo" in texto_norm:
        apetite = le_app.transform(["baixo"])[0]
    else:
        apetite = le_app.transform(["normal"])[0]

    if "sem andar" in texto_norm or "nao conseguindo ficar de estacao" in texto_norm:
        mobilidade = le_mob.transform(["sem andar"])[0]
    elif "limitada" in texto_norm or "fraqueza" in texto_norm:
        mobilidade = le_mob.transform(["limitada"])[0]
    else:
        mobilidade = le_mob.transform(["normal"])[0]

    doencas_detectadas = [d for d in palavras_chave_eutanasia if d in texto_norm]
    tem_doenca_letal = int(len(doencas_detectadas) > 0)

    dados = [[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]]
    dados_df = pd.DataFrame(dados, columns=features_eutanasia)

    # Debug: valores extraídos
    st.write("### Dados extraídos da anamnese:")
    st.write(dados_df)

    # Previsões modelos
    alta = modelo_alta.predict(dados_df[features])[0]
    internar = int(modelo_internar.predict(dados_df[features])[0])
    dias = int(round(modelo_dias.predict(dados_df[features])[0])) if internar == 1 else 0
    eutanasia_chance = round(modelo_eutanasia.predict_proba(dados_df)[0][1] * 100, 1)

    # Ajuste usando heurísticas
    alta_h, internar_h, dias_h, eutanasia_h = heuristicas_para_valores_reais(dados_df.iloc[0], le_mob, le_app)

    if internar_h > internar:
        internar = internar_h
        dias = dias_h

    if eutanasia_h > 0 and eutanasia_chance < 90:
        eutanasia_chance = max(eutanasia_chance, 90)

    # Ajuste para sintomas graves não detectados pelo modelo
    termos_graves = ["cancer", "terminal", "insuficiencia renal", "falencia multiple", "convulsao", "coma"]
    if doencas_detectadas and any(p in texto_norm for p in termos_graves):
        if eutanasia_chance < 90:
            eutanasia_chance = 95.0
    else:
        if dor >= 7 or apetite == le_app.transform(["nenhum"])[0] or mobilidade == le_mob.transform(["sem andar"])[0] or temperatura > 40 or gravidade == 10:
            eutanasia_chance = max(eutanasia_chance, 50.0)

    return {
        "Alta": "Sim" if alta == 1 else "Não",
        "Internar": "Sim" if internar == 1 else "Não",
        "Dias Internado": dias,
        "Chance de Eutanásia (%)": eutanasia_chance,
        "Doenças Detectadas": doencas_detectadas if doencas_detectadas else ["Nenhuma grave"]
    }

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

anamnese = st.text_area("Digite a anamnese do paciente:")

if st.button("Analisar"):
    if anamnese.strip() == "":
        st.warning("Por favor, digite a anamnese para análise.")
    else:
        resultado = prever(anamnese)
        st.subheader("📋 Resultado da Avaliação:")
        for chave, valor in resultado.items():
            if isinstance(valor, list):
                valor = ", ".join(valor)
            st.write(f"**{chave}**: {valor}")

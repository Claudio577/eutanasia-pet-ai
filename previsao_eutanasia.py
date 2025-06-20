import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier
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
    modelo_dias = RandomForestClassifier(random_state=42).fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias

def prever(texto):
    texto_norm = normalizar_texto(texto)

    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", texto_norm, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", texto_norm, float, 10.0)
    temperatura = extrair_variavel(r"(\d{2}(?:\.\d+)?)\s*(?:graus|c|celsius|\º c)", texto_norm, float, 38.5)
    gravidade = 10 if "vermelho" in texto_norm else (7 if "amarelo" in texto_norm else 5)

    if any(p in texto_norm for p in ["dor extrema", "dor 10", "dor dez", "dor intensa"]):
        dor = 10
    elif any(p in texto_norm for p in ["dor forte", "dor severa", "dor 8", "dor 9"]):
        dor = 8
    elif any(p in texto_norm for p in ["dor moderada", "dor 5", "dor 6", "dor 7"]):
        dor = 5
    elif any(p in texto_norm for p in ["sem dor", "ausencia de dor", "dor 0"]):
        dor = 0
    else:
        dor = 3

    if any(p in texto_norm for p in ["nenhum apetite", "não come", "recusa alimentar", "anorexia completa"]):
        apetite = le_app.transform(["nenhum"])[0]
    elif any(p in texto_norm for p in ["baixo apetite", "apetite baixo", "pouco apetite", "come pouco"]):
        apetite = le_app.transform(["baixo"])[0]
    else:
        apetite = le_app.transform(["normal"])[0]

    if any(p in texto_norm for p in ["sem andar", "não consegue andar", "decúbiso permanente", "paralisia"]):
        mobilidade = le_mob.transform(["sem andar"])[0]
    elif any(p in texto_norm for p in ["mobilidade limitada", "dificuldade para andar", "anda pouco", "fraqueza"]):
        mobilidade = le_mob.transform(["limitada"])[0]
    else:
        mobilidade = le_mob.transform(["normal"])[0]

    doencas_detectadas = [d for d in palavras_chave_eutanasia if d in texto_norm]
    tem_doenca_letal = int(len(doencas_detectadas) > 0)

    dados = [[idade, peso, gravidade, dor, mobilidade, apetite, temperatura, tem_doenca_letal]]
    dados_df = pd.DataFrame(dados, columns=features_eutanasia)

    alta = modelo_alta.predict(dados_df[features])[0]
    internar = int(modelo_internar.predict(dados_df[features])[0])
    dias = int(modelo_dias.predict(dados_df[features])[0]) if internar == 1 else 0
    eutanasia_chance = round(modelo_eutanasia.predict_proba(dados_df)[0][1] * 100, 1)

    condicoes_graves = (
        dor >= 8 or 
        apetite == le_app.transform(["nenhum"])[0] or 
        mobilidade == le_mob.transform(["sem andar"])[0] or 
        temperatura > 39.5 or 
        gravidade == 10
    )

    if len(doencas_detectadas) >= 2 and condicoes_graves:
        eutanasia_chance = max(eutanasia_chance, 95.0)
    elif doencas_detectadas and condicoes_graves:
        eutanasia_chance = max(eutanasia_chance, 90.0)
    elif doencas_detectadas or condicoes_graves:
        eutanasia_chance = max(eutanasia_chance, 75.0)
    elif dor >= 8:
        eutanasia_chance = max(eutanasia_chance, 60.0)

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
    normalizar_texto(d) for d in df_doencas['Doença'].dropna().unique()
] + [
    "pancreatite aguda",
    "falencia hepatica",
    "insuficiencia renal",
    "septicemia",
    "peritonite"
]

le_mob = LabelEncoder()
le_app = LabelEncoder()
df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())

df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
    lambda d: int(any(p in normalizar_texto(d) for p in palavras_chave_eutanasia))
)

features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
features_eutanasia = features + ['tem_doenca_letal']

modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias = treinar_modelos(df, le_mob, le_app)

# =======================
# INTERFACE STREAMLIT
# =======================
st.title("📉 Avaliação Clínica Canina")

anamnese = st.text_area("Digite a anamnese do paciente:")

if st.button("Analisar"):
    resultado = prever(anamnese)
    st.subheader("📋 Resultado da Avaliação:")
    st.write(f"**Alta**: {resultado['Alta']}")
    st.write(f"**Internar**: {resultado['Internar']}")
    st.write(f"**Dias Internado**: {resultado['Dias Internado']}")
    st.write(f"**Chance de Eutanásia (%)**: {resultado['Chance de Eutanásia (%)']}")
    st.write(f"**Doenças Detectadas**: {', '.join(resultado['Doenças Detectadas'])}")

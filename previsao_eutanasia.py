import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ----------------------
# FUNÇÕES AUXILIARES
# ----------------------
def normalizar_texto(texto):
    return re.sub(r'\s+', ' ',
                  re.sub(r'[^\w\s]', ' ',
                         unicodedata.normalize('NFKD', texto)
                         .encode('ASCII', 'ignore')
                         .decode('utf-8')
                         .lower()
                  ).strip()
    )

def extrair_variavel(padrao, texto, tipo=float, valor_padrao=None):
    m = re.search(padrao, texto)
    return tipo(m.group(1)) if (m and m.group(1)) else valor_padrao

def heuristicas_para_valores_reais(linha, le_mob, le_app):
    dor = linha['Dor']; temp = linha['Temperatura']
    mobilidade = linha['Mobilidade']; apetite = linha['Apetite']
    letal = linha['tem_doenca_letal']

    alta = int(not letal and dor <= 4 and 37.5 <= temp <= 39
               and mobilidade == le_mob.transform(['normal'])[0]
               and apetite == le_app.transform(['normal'])[0])
    internar = int(dor >= 7 or temp > 39.5 or temp < 37
                   or mobilidade in le_mob.transform(['sem andar', 'limitada'])
                   or apetite in le_app.transform(['nenhum', 'baixo']))
    dias = 0 if not internar else (7 if letal or dor >= 8 or temp > 40 else 3)
    eutanasia = int(letal or dor >= 9 or apetite == le_app.transform(['nenhum'])[0]
                    or mobilidade == le_mob.transform(['sem andar'])[0]
                    or temp > 40)
    return alta, internar, dias, eutanasia

# ----------------------
# FUNÇÃO DE APLICAÇÃO SMOTE COM TRATAMENTO DE ERRO
# ----------------------
def aplicar_smote(X, y, label=""):
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    except Exception as e:
        st.warning(f"⚠️ SMOTE não aplicado para {label}: {e}")
        return X, y

# ----------------------
# TREINAMENTO DE MODELOS
# ----------------------
def treinar_modelos(df, features, features_eutanasia, le_mob, le_app):
    # Eutanásia
    X_e = df[features_eutanasia].fillna(df[features_eutanasia].mean(numeric_only=True)).astype(float)
    y_e = df['Eutanasia'].fillna(df['Eutanasia'].mode()[0]).astype(int)
    Xe_tr, _, ye_tr, _ = train_test_split(X_e, y_e, test_size=0.2, random_state=42, stratify=y_e)
    Xe_res, ye_res = aplicar_smote(Xe_tr, ye_tr, label="Eutanásia")
    model_e = RandomForestClassifier(class_weight='balanced', random_state=42)
    model_e.fit(Xe_res, ye_res)

    # Alta
    X_a = df[features].fillna(df[features].mean(numeric_only=True)).astype(float)
    y_a = df['Alta'].fillna(df['Alta'].mode()[0]).astype(int)
    Xa_tr, _, ya_tr, _ = train_test_split(X_a, y_a, test_size=0.2, random_state=42, stratify=y_a)
    Xa_res, ya_res = aplicar_smote(Xa_tr, ya_tr, label="Alta")
    model_a = RandomForestClassifier(class_weight='balanced', random_state=42)
    model_a.fit(Xa_res, ya_res)

    # Internar
    model_i = RandomForestClassifier(random_state=42)
    model_i.fit(X_a, df['Internar'].fillna(0).astype(int))

    # Dias Internado
    dias_df = df[df['Internar'] == 1].copy()
    dias_df['Dias Internado'] = dias_df['Dias Internado'].fillna(0).astype(int)
    model_d = RandomForestClassifier(random_state=42)
    model_d.fit(dias_df[features], dias_df['Dias Internado'])

    return model_e, model_a, model_i, model_d

# ----------------------
# FUNÇÃO DE PREVISÃO
# ----------------------
def prever(texto):
    tn = normalizar_texto(texto)
    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", tn, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", tn, float, 10.0)
    temp = extrair_variavel(r"(\d{2}(?:\.\d+)?)\s*(?:graus|c|celsius|ºc)", tn, float, 38.5)
    grav = 10 if "vermelho" in tn else 5

    dor = 10 if "dor intensa" in tn else 5 if "dor moderada" in tn else 0 if "sem dor" in tn else 4
    ap = 'nenhum' if "nenhum apetite" in tn else 'baixo' if "baixo apetite" in tn or "apetite baixo" in tn else 'normal'
    mob = 'sem andar' if "sem andar" in tn or "nao conseguindo ficar de estacao" in tn else \
          'limitada' if "limitada" in tn or "fraqueza" in tn else 'normal'

    ap = le_app.transform([ap])[0]
    mob = le_mob.transform([mob])[0]
    letal = int(any(p in tn for p in palavras_chave_eutanasia))

    df_pre = pd.DataFrame([[idade, peso, grav, dor, mob, ap, temp, letal]],
                          columns=features_eutanasia)

    out = {}
    out['Alta'] = "Sim" if model_alta.predict(df_pre[features])[0] else "Não"
    out['Internar'] = "Sim" if model_internar.predict(df_pre[features])[0] else "Não"
    out['Dias Internado'] = int(model_dias.predict(df_pre[features])[0] if out['Internar'] == "Sim" else 0)
    chance = model_eutanasia.predict_proba(df_pre)[0][1] * 100
    out['Chance de Eutanásia (%)'] = round(chance, 1)
    out['Doenças Detectadas'] = [d for d in palavras_chave_eutanasia if d in tn] or ["Nenhuma grave"]

    return out

# ----------------------
# CARREGAMENTO DE DADOS E TREINO
# ----------------------
try:
    df = pd.read_csv("Casos_Cl_nicos_Simulados.csv")
    df_do = pd.read_csv("doencas_caninas_eutanasia_expandidas.csv")
except FileNotFoundError as e:
    st.error(f"CSV não encontrado: {e.filename}")
    st.stop()

palavras_chave_eutanasia = [
    unicodedata.normalize('NFKD', d).encode('ASCII', 'ignore').decode('utf-8').lower().strip()
    for d in df_do['Doença'].dropna().unique()
]

le_mob = LabelEncoder()
le_app = LabelEncoder()
df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())
df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
    lambda d: int(any(p in unicodedata.normalize('NFKD', d)
                       .encode('ASCII', 'ignore').decode('utf-8')
                       .lower() for p in palavras_chave_eutanasia))
)

features = ['Idade', 'Peso', 'Gravidade', 'Dor', 'Mobilidade', 'Apetite', 'Temperatura']
features_eutanasia = features + ['tem_doenca_letal']

model_eutanasia, model_alta, model_internar, model_dias = \
    treinar_modelos(df, features, features_eutanasia, le_mob, le_app)

# ----------------------
# INTERFACE STREAMLIT
# ----------------------
st.title("💉 Avaliação Clínica Canina")
texto = st.text_area("Digite a anamnese:")

if st.button("Analisar"):
    res = prever(texto)
    st.subheader("📋 Resultado")
    for k, v in res.items():
        st.write(f"**{k}**: {v if not isinstance(v, list) else ', '.join(v)}")


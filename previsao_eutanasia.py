import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

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

# ----------------------
# TREINAMENTO DE MODELOS
# ----------------------
def treinar_modelos(df, features, features_eutanasia, le_mob, le_app):
    cat_indices = [features_eutanasia.index('Mobilidade'), features_eutanasia.index('Apetite')]
    smote = SMOTENC(categorical_features=cat_indices, random_state=42)

    # ---------- EUTANÁSIA ----------
    X_e = df[features_eutanasia].fillna(df[features_eutanasia].mean(numeric_only=True)).astype(float).to_numpy()
    y_e = df['Eutanasia'].fillna(df['Eutanasia'].mode()[0]).astype(int).to_numpy()

    st.write(f"Distribuição Eutanásia: {np.bincount(y_e)}")
    st.write(f"NaNs em X_e: {np.isnan(X_e).sum()}, NaNs em y_e: {np.isnan(y_e).sum()}")

    if len(np.unique(y_e)) < 2 or min(np.bincount(y_e)) < 2:
        st.warning("Dados insuficientes para aplicar SMOTENC na predição de eutanásia.")
        model_e = RandomForestClassifier(class_weight='balanced', random_state=42)
        model_e.fit(X_e, y_e)
    else:
        Xe_tr, _, ye_tr, _ = train_test_split(X_e, y_e, test_size=0.2, random_state=42, stratify=y_e)

        Xe_tr = np.array(Xe_tr, dtype=float)
        ye_tr = np.array(ye_tr, dtype=int).ravel()

        Xe_res, ye_res = smote.fit_resample(Xe_tr, ye_tr)

        model_e = RandomForestClassifier(class_weight='balanced', random_state=42)
        model_e.fit(Xe_res, ye_res)

    # ---------- ALTA ----------
    X_a = df[features].fillna(df[features].mean(numeric_only=True)).astype(float).to_numpy()
    y_a = df['Alta'].fillna(df['Alta'].mode()[0]).astype(int).to_numpy()

    if len(np.unique(y_a)) < 2 or min(np.bincount(y_a)) < 2:
        st.warning("Dados insuficientes para aplicar SMOTENC na predição de alta.")
        model_a = RandomForestClassifier(class_weight='balanced', random_state=42)
        model_a.fit(X_a, y_a)
    else:
        Xa_tr, _, ya_tr, _ = train_test_split(X_a, y_a, test_size=0.2, random_state=42, stratify=y_a)
        Xa_tr = np.array(Xa_tr, dtype=float)
        ya_tr = np.array(ya_tr, dtype=int).ravel()
        Xa_res, ya_res = smote.fit_resample(Xa_tr, ya_tr)

        model_a = RandomForestClassifier(class_weight='balanced', random_state=42)
        model_a.fit(Xa_res, ya_res)

    # ---------- INTERNAR ----------
    model_i = RandomForestClassifier(random_state=42)
    model_i.fit(X_a, df['Internar'].astype(int).values)

    # ---------- DIAS INTERNADO ----------
    df_internado = df[df['Internar'] == 1]
    if df_internado.shape[0] > 0:
        model_d = RandomForestClassifier(random_state=42)
        model_d.fit(df_internado[features].fillna(df_internado[features].mean(numeric_only=True)).astype(float).values,
                    df_internado['Dias Internado'].astype(int).values)
    else:
        model_d = None
        st.warning("Sem dados para treinar modelo de Dias Internado.")

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

    df_pre = pd.DataFrame([[idade,peso,grav,dor,mob,ap,temp,letal]],
                          columns=features_eutanasia)

    out = {}
    out['Alta'] = "Sim" if model_alta.predict(df_pre[features])[0] else "Não"
    out['Internar'] = "Sim" if model_internar.predict(df_pre[features])[0] else "Não"
    out['Dias Internado'] = int(model_dias.predict(df_pre[features])[0] if out['Internar']=="Sim" and model_dias else 0)
    chance = model_eutanasia.predict_proba(df_pre[features_eutanasia])[0][1] * 100
    out['Chance de Eutanásia (%)'] = round(chance,1)
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
    unicodedata.normalize('NFKD', d).encode('ASCII','ignore').decode('utf-8').lower().strip()
    for d in df_do['Doença'].dropna().unique()
]

le_mob = LabelEncoder()
le_app = LabelEncoder()
df['Mobilidade'] = le_mob.fit_transform(df['Mobilidade'].str.lower().str.strip())
df['Apetite'] = le_app.fit_transform(df['Apetite'].str.lower().str.strip())
df['tem_doenca_letal'] = df['Doença'].fillna("").apply(
    lambda d: int(any(p in unicodedata.normalize('NFKD', d)
                       .encode('ASCII','ignore').decode('utf-8')
                       .lower() for p in palavras_chave_eutanasia))
)

features = ['Idade','Peso','Gravidade','Dor','Mobilidade','Apetite','Temperatura']
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
    for k,v in res.items():
        st.write(f"**{k}**: {v if not isinstance(v,list) else ', '.join(v)}")



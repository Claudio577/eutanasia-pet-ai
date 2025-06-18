import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y
from imblearn.over_sampling import SMOTE

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

def treinar_modelo_com_smote(X, y, random_state=42):
    smote = SMOTE(random_state=random_state)
    # Verificar número de classes e quantidade mínima para aplicar SMOTE
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        # Só uma classe, treinar direto sem SMOTE
        model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        model.fit(X, y)
        return model
    elif np.min(counts) < 2:
        # Classes com menos de 2 amostras, treinar direto sem SMOTE
        model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        model.fit(X, y)
        return model
    else:
        X_res, y_res = smote.fit_resample(X, y)
        model = RandomForestClassifier(class_weight='balanced', random_state=random_state)
        model.fit(X_res, y_res)
        return model

def treinar_modelos(df, features, features_eutanasia, le_mob, le_app):
    # Preparar dados para Eutanásia
    X_e = df[features_eutanasia].fillna(df[features_eutanasia].mean()).astype(float)
    y_e = df['Eutanasia'].fillna(df['Eutanasia'].mode()[0]).astype(int)
    X_e, y_e = check_X_y(X_e, y_e, accept_sparse=False)
    Xe_tr, _, ye_tr, _ = train_test_split(X_e, y_e, test_size=0.2,
                                          random_state=42, stratify=y_e)
    model_e = treinar_modelo_com_smote(Xe_tr, ye_tr)

    # Preparar dados para Alta
    X_a = df[features].fillna(df[features].mean()).astype(float)
    y_a = df['Alta'].fillna(df['Alta'].mode()[0]).astype(int)
    X_a, y_a = check_X_y(X_a, y_a, accept_sparse=False)
    Xa_tr, _, ya_tr, _ = train_test_split(X_a, y_a, test_size=0.2,
                                          random_state=42, stratify=y_a)
    model_a = treinar_modelo_com_smote(Xa_tr, ya_tr)

    # Internar
    model_i = RandomForestClassifier(random_state=42)
    model_i.fit(X_a, df['Internar'])

    # Dias internado (somente casos internados)
    df_internar = df[df['Internar'] == 1]
    if len(df_internar) > 0:
        model_d = RandomForestClassifier(random_state=42)
        model_d.fit(df_internar[features].fillna(df[features].mean()).astype(float),
                    df_internar['Dias Internado'])
    else:
        model_d = None

    return model_e, model_a, model_i, model_d

# Carregamento dados e LabelEncoder (mantém igual ao seu código)
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

model_eutanasia, model_alta, model_internar, model_dias = treinar_modelos(
    df, features, features_eutanasia, le_mob, le_app
)

# Função prever e interface Streamlit seguem iguais (se quiser, mando de novo)



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
        apetite in [le_app.transform(['nenhum'])[0], le_app.transform(['baixo'])[0]] or
        tem_doenca_letal == 1
    ) else 0

    if internar == 0:
        dias = 0
    else:
        dias = 7 if tem_doenca_letal == 1 or dor >= 8 or temp > 40 or mobilidade == le_mob.transform(['sem andar'])[0] else 3

    eutanasia = 1 if (
        tem_doenca_letal == 1 or
        dor >= 8 or
        apetite == le_app.transform(['nenhum'])[0] or
        mobilidade == le_mob.transform(['sem andar'])[0] or
        temp > 39.5 or
        (tem_doenca_letal == 1 and dor >= 5)
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
    modelo_dias = RandomForestClassifier(random_state=42).fit(df[df['Internar'] == 1][features], df[df['Internar'] == 1]['Dias Internado'])

    return modelo_eutanasia, modelo_alta, modelo_internar, modelo_dias

def prever(texto):
    texto_norm = normalizar_texto(texto)

    idade = extrair_variavel(r"(\d+(?:\.\d+)?)\s*anos?", texto_norm, float, 5.0)
    peso = extrair_variavel(r"(\d+(?:\.\d+)?)\s*kg", texto_norm, float, 10.0)
    temperatura = extrair_variavel(r"(\d{2}(?:\.\d+)?)\s*(?:graus|c|celsius|ºc)", texto_norm, float, 38.5)
    gravidade = 10 if "vermelho" in texto_norm else 5

    if "dor intensa" in texto_norm or "dor extrema" in texto_norm:
        dor = 10
    elif "dor moderada" in texto_norm:
        dor = 5
    elif "sem dor" in texto_norm:
        dor = 0
    else:
        dor = 4

    if "nenhum apetite" in texto_norm or "não come" in texto_norm or "recusa alimentar" in texto_norm:
        apetite = le_app.transform(["nenhum"])[0]
    elif "baixo apetite" in texto_norm or "apetite baixo" in texto_norm or "pouco apetite" in texto_norm:
        apetite = le_app.transform(["baixo"])[0]
    else:
        apetite = le_app.transform(["normal"])[0]

    if "sem andar" in texto_norm or "não consegue ficar de pé" in texto_norm or "paralisia" in texto_norm:
        mobilidade = le_mob.transform(["sem andar"])[0]
    elif "limitada" in texto_norm or "fraqueza" in texto_norm or "dificuldade para andar" in texto_norm:
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

    # Lógica aprimorada para cálculo da chance de eutanásia
    condicoes_graves = (
        doencas_detectadas or 
        dor >= 9 or 
        apetite == le_app.transform(["nenhum"])[0] or 
        mobilidade == le_mob.transform(["sem andar"])[0] or 
        temperatura > 39.5 or 
        gravidade == 10
    )
    
    if doencas_detectadas and condicoes_graves:
        eutanasia_chance = 95.0
    elif doencas_detectadas or condicoes_graves:
        eutanasia_chance = max(eutanasia_chance, 75.0)
    elif dor >= 9:
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
st.title("💉 Avaliação Clínica Canina")

anamnese = st.text_area("Digite a anamnese do paciente:")

if st.button("Analisar"):
    resultado = prever(anamnese)
    st.subheader("📋 Resultado da Avaliação:")
    st.write(f"**Alta**: {resultado['Alta']}")
    st.write(f"**Internar**: {resultado['Internar']}")
    st.write(f"**Dias Internado**: {resultado['Dias Internado']}")
    st.write(f"**Chance de Eutanásia (%)**: {resultado['Chance de Eutanásia (%)']}")
    st.write(f"**Doenças Detectadas**: {', '.join(resultado['Doenças Detectadas'])}")

# Seção de exemplos
st.header("📋 Exemplos de Anamneses para Teste")

exemplo_alta = """
Cachorro da raça Labrador, 4 anos, 32 kg, apresentando-se em bom estado geral. 
O proprietário relata que o animal está ativo, com apetite normal e sem sinais de dor. 
Exame físico revela temperatura de 38.2°C, mucosas rosadas e hidratadas. 
O animal caminha normalmente e não apresenta alterações significativas ao exame.
"""

exemplo_eutanasia_sem_doenca = """
Cão idoso, 12 anos, 18 kg, apresentando-se em mau estado geral. 
Proprietário relata que o animal não consegue mais se levantar, não come há 3 dias 
e apresenta dor intensa. Temperatura de 37.0°C. 
O animal apresenta caquexia e desidratação severa. 
Não foi diagnosticada nenhuma doença terminal específica.
"""

exemplo_eutanasia_com_doenca = """
Cão da raça Pastor Alemão, 8 anos, 35 kg, diagnosticado com osteossarcoma metastático. 
Apresenta dor extrema (10/10), não consegue se levantar, apetite nenhum nos últimos 5 dias. 
Temperatura de 39.8°C. O tumor é inoperável e já apresenta metástases pulmonares. 
O animal está em sofrimento constante mesmo com medicação analgésica máxima.
"""

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Caso para Alta")
    st.text_area("Anamnese 1", exemplo_alta, height=200, key="ex1")
    if st.button("Testar Caso 1"):
        resultado = prever(exemplo_alta)
        st.write(resultado)

with col2:
    st.subheader("Eutanásia (Sem Doença)")
    st.text_area("Anamnese 2", exemplo_eutanasia_sem_doenca, height=200, key="ex2")
    if st.button("Testar Caso 2"):
        resultado = prever(exemplo_eutanasia_sem_doenca)
        st.write(resultado)

with col3:
    st.subheader("Eutanásia (Com Doença)")
    st.text_area("Anamnese 3", exemplo_eutanasia_com_doenca, height=200, key="ex3")
    if st.button("Testar Caso 3"):
        resultado = prever(exemplo_eutanasia_com_doenca)
        st.write(resultado)

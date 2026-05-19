import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="LLM Biology Evaluator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1a73e8;
        text-align: center;
        padding: 1rem 0;
    }
    .response-box {
        background: #f8f9fa;
        border-left: 4px solid #1a73e8;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .green-box  { border-left-color: #2ecc71 !important; }
    .orange-box { border-left-color: #f39c12 !important; }
    .red-box    { border-left-color: #e74c3c !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🧬 Évaluateur de Robustesse LLM en Biologie</div>',
    unsafe_allow_html=True
)
st.caption("Projet académique — Licence — Évaluation de la robustesse des LLM aux paraphrases")
st.divider()

# ============================================================
# BARRE LATÉRALE
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    model_choice = st.selectbox(
        "🤖 Modèle LLM",
        ["gemini", "deepseek", "flan-t5-small"],
        help="Gemini et DeepSeek utilisent les clés API"
    )

    theme_choice = st.selectbox(
        "🔬 Thème biologique",
        ["général", "ADN", "ARN", "Cellule", "Génétique",
         "Photosynthèse", "Respiration cellulaire",
         "Immunologie", "Microbiologie"]
    )

    st.divider()
    st.subheader("📝 Questions exemples")

    exemples = [
        "Qu'est-ce que l'ADN ?",
        "Où se déroule la photosynthèse ?",
        "Quel est le rôle des mitochondries ?",
        "Quelle est la différence entre ADN et ARN ?",
        "Comment les anticorps fonctionnent-ils ?",
        "Combien de chromosomes a une cellule humaine ?",
        "Quels sont les types d'ARN ?",
    ]

    selected_example = st.radio(
        "Choisir un exemple :",
        exemples,
        index=None
    )

    st.divider()
    st.info("💡 Ce projet teste si le LLM donne des réponses cohérentes quand on reformule une question.")

# ============================================================
# ONGLETS
# ============================================================
tab1, tab2, tab3 = st.tabs(["🔬 Évaluation", "📊 Historique", "📈 Statistiques"])

# ──────────────────────────────────────
# ONGLET 1 — ÉVALUATION
# ──────────────────────────────────────
with tab1:
    st.header("🔬 Tester la Robustesse")

    question = st.text_area(
        "✏️ Entrez votre question de biologie :",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="Ex: Qu'est-ce que la photosynthèse ?"
    )

    st.markdown("**🔄 Paraphrases personnalisées (optionnel) :**")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        custom_para_syn = st.text_area(
            "🟡 Paraphrase syntaxique :",
            height=80,
            placeholder="Ex: Quelle est la définition de l'ADN ?"
        )
    with col_p2:
        custom_para_lex = st.text_area(
            "🟢 Paraphrase lexicale :",
            height=80,
            placeholder="Ex: Qu'est-ce que la molécule d'ADN ?"
        )
    st.caption("💡 Laissez vide pour génération automatique.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        btn = st.button("🚀 Lancer l'évaluation", use_container_width=True)

    if btn and question:
        with st.spinner(f"⏳ {model_choice.upper()} analyse en cours..."):
            try:
                payload = {
                    "question": question,
                    "model": model_choice,
                    "theme": theme_choice
                }
                if custom_para_syn:
                    payload["custom_para_syn"] = custom_para_syn
                if custom_para_lex:
                    payload["custom_para_lex"] = custom_para_lex

                response = requests.post(
                    f"{API_URL}/evaluate",
                    json=payload,
                    timeout=180
                )

                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Évaluation terminée avec {model_choice.upper()} !")
                    st.divider()

                    # Scores
                    st.subheader("📊 Scores de Robustesse")
                    c1, c2, c3 = st.columns(3)

                    s_syn = data['sim_syntactic']
                    s_lex = data['sim_lexical']
                    s_rob = data['robustness_score']

                    def emoji_score(s):
                        if s >= 0.85: return "🟢"
                        elif s >= 0.65: return "🟡"
                        else: return "🔴"

                    c1.metric(
                        f"{emoji_score(s_syn)} Similarité syntaxique",
                        f"{s_syn:.3f}"
                    )
                    c2.metric(
                        f"{emoji_score(s_lex)} Similarité lexicale",
                        f"{s_lex:.3f}"
                    )
                    c3.metric(
                        f"{emoji_score(s_rob)} Score robustesse",
                        f"{s_rob:.3f}",
                        delta=data['classification']
                    )

                    # Jauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=s_rob * 100,
                        title={'text': "Robustesse (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#1a73e8"},
                            'steps': [
                                {'range': [0, 65],   'color': '#ffcccc'},
                                {'range': [65, 85],  'color': '#fff3cc'},
                                {'range': [85, 100], 'color': '#ccffcc'}
                            ],
                        }
                    ))
                    fig_gauge.update_layout(height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    st.divider()

                    # Paraphrases
                    st.subheader("🔄 Paraphrases Générées")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("**🔵 Question originale**")
                        st.info(data['question'])
                        st.markdown("**🟡 Paraphrase syntaxique**")
                        st.warning(data['paraphrase_syntaxique'])
                    with col_b:
                        st.markdown("**🟢 Paraphrase lexicale**")
                        st.success(data['paraphrase_lexicale'])

                    st.divider()

                    # Réponses
                    st.subheader(f"🤖 Réponses de {model_choice.upper()}")
                    st.markdown("**📌 Réponse à la question originale :**")
                    st.markdown(
                        f'<div class="response-box">{data["response_original"]}</div>',
                        unsafe_allow_html=True
                    )

                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.markdown(f"**🟡 Réponse syntaxique** — sim: {s_syn:.3f}")
                        css = "green-box" if s_syn >= 0.85 else ("orange-box" if s_syn >= 0.65 else "red-box")
                        st.markdown(
                            f'<div class="response-box {css}">{data["response_syntactic"]}</div>',
                            unsafe_allow_html=True
                        )
                    with col_r2:
                        st.markdown(f"**🟢 Réponse lexicale** — sim: {s_lex:.3f}")
                        css = "green-box" if s_lex >= 0.85 else ("orange-box" if s_lex >= 0.65 else "red-box")
                        st.markdown(
                            f'<div class="response-box {css}">{data["response_lexical"]}</div>',
                            unsafe_allow_html=True
                        )

                else:
                    st.error(f"❌ Erreur API : {response.status_code}")
                    st.code(response.text)

            except requests.ConnectionError:
                st.error("❌ Backend non accessible. Lance d'abord FastAPI !")

    elif btn and not question:
        st.warning("⚠️ Entrez une question d'abord.")

# ──────────────────────────────────────
# ONGLET 2 — HISTORIQUE
# ──────────────────────────────────────
with tab2:
    st.header("📊 Historique des Évaluations")

    if st.button("🔄 Actualiser"):
        st.rerun()

    try:
        resp = requests.get(f"{API_URL}/history?limit=50", timeout=10)
        if resp.status_code == 200:
            history = resp.json()["history"]
            if history:
                df_hist = pd.DataFrame(history)
                st.dataframe(
                    df_hist[['id', 'question', 'robustness_score',
                              'sim_syntactic', 'sim_lexical',
                              'model_used', 'timestamp']],
                    use_container_width=True
                )
                fig = px.line(
                    df_hist.sort_values('id'),
                    x='id',
                    y='robustness_score',
                    color='model_used',
                    title='Évolution du score de robustesse par modèle',
                    markers=True,
                    labels={'id': 'Évaluation n°',
                            'robustness_score': 'Score',
                            'model_used': 'Modèle'}
                )
                fig.add_hline(y=0.85, line_dash="dash",
                              line_color="green",
                              annotation_text="Seuil cohérence")
                fig.add_hline(y=0.65, line_dash="dash",
                              line_color="orange",
                              annotation_text="Seuil contradiction")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ℹ️ Aucune évaluation encore effectuée.")
    except:
        st.warning("⚠️ Lance le backend FastAPI pour voir l'historique.")

# ──────────────────────────────────────
# ONGLET 3 — STATISTIQUES
# ──────────────────────────────────────
with tab3:
    st.header("📈 Statistiques Globales")

    try:
        resp = requests.get(f"{API_URL}/stats", timeout=10)
        if resp.status_code == 200:
            stats = resp.json()

            if "message" not in stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total évaluations",
                          stats['total_evaluations'])
                c2.metric("Robustesse moyenne",
                          f"{stats['avg_robustness_score']:.3f}")
                c3.metric("Sim. syntaxique moy.",
                          f"{stats['avg_sim_syntactic']:.3f}")
                c4.metric("Sim. lexicale moy.",
                          f"{stats['avg_sim_lexical']:.3f}")

                fig_bar = go.Figure(go.Bar(
                    x=['Robustesse globale',
                       'Paraphrase syntaxique',
                       'Paraphrase lexicale'],
                    y=[stats['avg_robustness_score'],
                       stats['avg_sim_syntactic'],
                       stats['avg_sim_lexical']],
                    marker_color=['#1a73e8', '#34a853', '#fbbc04'],
                    text=[f"{v:.3f}" for v in [
                        stats['avg_robustness_score'],
                        stats['avg_sim_syntactic'],
                        stats['avg_sim_lexical']
                    ]],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    title="Scores moyens de robustesse",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                fig_bar.add_hline(y=0.85, line_dash="dash",
                                  line_color="green")
                fig_bar.add_hline(y=0.65, line_dash="dash",
                                  line_color="orange")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("ℹ️ Lance quelques évaluations d'abord.")
    except:
        st.warning("⚠️ Lance le backend FastAPI pour voir les stats.")
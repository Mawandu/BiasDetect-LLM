import streamlit as st
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.linguistic import LinguisticAnalyzer
from modules.symbolic import SymbolicAnalyzer
from modules.fusion import FusionEngine

# Page configuration
st.set_page_config(
    page_title="BiasDetect-LLM",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_analyzers(model_name):
    ling = LinguisticAnalyzer(model_name)
    sym = SymbolicAnalyzer() # Default en_core_web_sm
    fusion = FusionEngine()
    return ling, sym, fusion

# Title and introduction
st.title("BiasDetect-LLM")
st.markdown("""
**Système Hybride de Détection de Biais**
Ce démonstrateur analyse les textes générés par des LLM pour détecter des biais de genre, culturels et de représentation en utilisant une approche hybride (Neuronal + Symbolique).
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    model_choice = st.selectbox(
        "Modèle d'Embeddings",
        ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
    )
    alpha = st.slider(
        "Pondération (Alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        help="0 = 100% Symbolique, 1 = 100% Neuronal. Plus alpha est élevé, plus le modèle neuronal domine."
    )
    
    st.info("Assurez-vous d'avoir installé les dépendances (`pip install -r requirements.txt`) et le modèle spacy (`python -m spacy download en_core_web_sm`).")

# Load analyzers
try:
    ling_analyzer, sym_analyzer, fusion_engine = load_analyzers(model_choice)
except Exception as e:
    st.error(f"Erreur au chargement des modèles : {e}")
    st.stop()

# Main Input Area
col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area(
        "Texte à analyser",
        height=300,
        placeholder="Collez ici le texte généré par un LLM (ex: 'The doctor called the nurse and asked her for the file...')"
    )
    analyze_btn = st.button("Analyser", type="primary")

with col2:
    st.info("**Conseil** : Essayez avec des textes de descriptions de postes ('Search for a ninja developer') ou des histoires générées par IA.")

# Placeholder for results
if analyze_btn and user_input:
    st.divider()
    st.subheader("Résultats de l'analyse")
    
    # Calculate results first inside spinner
    with st.spinner("Analyse hybride en cours..."):
        # 1. Linguistic Analysis
        ling_res = ling_analyzer.analyze_text(user_input)
        
        # 2. Symbolic Analysis
        sym_res = sym_analyzer.analyze_text(user_input)
        
        # 3. Fusion
        fusion_res = fusion_engine.fuse_scores(ling_res, sym_res, alpha)
        
    # Render results outside spinner (Fixes 'ElementNode' error)
    final_score = fusion_res['final_score']
    
    # Display main metric
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        st.metric("Score Global de Biais (0-100)", f"{final_score}/100")
        st.progress(float(final_score) / 100)
    
    with col_metric2:
        st.write(fusion_res['explanation'])
    
    tab1, tab2, tab3 = st.tabs(["📊 Vue Détaillée", "🧠 Neuronal (Linguistique)", "📚 Symbolique (Ontologique)"])
    
    with tab1:
        st.markdown("### Synthèse")
        st.json(fusion_res)
        
    with tab2:
        st.markdown("### Analyse Linguistique")
        st.write(f"Direction estimée : **{ling_res.get('direction')}**")
        st.write(f"Score brut (subspace projection) : {ling_res.get('score'):.2f}")
        with st.expander("Voir détails techniques"):
            st.write(ling_res)
        
    with tab3:
        st.markdown("### Analyse Symbolique")
        issues = sym_res.get('issues', [])
        if not issues:
            st.success("Aucune violation ontologique flagrante détectée.")
        else:
            for issue in issues:
                st.warning(f"**[{issue.get('type', 'Biais')}]** {issue['term']} → **Stéréotype:** {issue['stereotype_risk']}")
                st.caption(issue['explanation'])


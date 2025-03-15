import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import time

# ----- Configuration de la page -----
st.set_page_config(
    page_title="MédiSoin - Portail Patient 💗",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- CSS personnalisé - Thème médical féminin -----
st.markdown("""
<style>
    /* Palette de couleurs féminines et médicales */
    :root {
        --primary: #FF69B4;
        --primary-light: #FFB6C1;
        --secondary: #9370DB;
        --accent: #F06292;
        --background: #FFF0F5;
        --text: #6A0DAD;
        --success: #8BC34A;
        --info: #29B6F6;
    }
    
    /* Fond et style général */
    .main {
        background-color: var(--background);
        background-image: linear-gradient(135deg, #FFF0F5 0%, #F8F8FF 100%);
        color: var(--text);
        font-family: 'Quicksand', sans-serif;
    }
    
    /* En-têtes */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        background: linear-gradient(90deg, #FF69B4 0%, #9370DB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: var(--secondary);
        text-align: center;
    }
    
    /* Cartes */
    .card {
        padding: 1.5rem;
        border-radius: 1rem;
        background-color: white;
        box-shadow: 0 0.3rem 0.8rem rgba(255, 105, 180, 0.15);
        transition: transform 0.3s ease;
        border-top: 4px solid var(--primary);
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Carte pour patient */
    .patient-card {
        background-color: white;
        border-left: 4px solid var(--primary);
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 0.7rem;
    }
    
    /* Carte pour docteur */
    .doctor-card {
        background-color: #F5F5FF;
        border-left: 4px solid var(--secondary);
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 0.7rem;
    }
    
    /* Métriques */
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--primary);
    }
    
    .metric-label {
        font-size: 1rem;
        color: var(--text);
        opacity: 0.8;
    }
    
    /* Boutons */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border: none;
        border-radius: 2rem;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: var(--secondary);
        transform: scale(1.02);
    }
    
    /* Séparateurs */
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255, 105, 180, 0), rgba(255, 105, 180, 0.5), rgba(255, 105, 180, 0));
        margin: 1.5rem 0;
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        background-color: #FFF0F5;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        color: white;
    }
    
    /* Uploader */
    .uploadedFile {
        border: 1px dashed var(--primary);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        color: var(--primary);
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 1px solid var(--primary-light);
    }
    
    .stSelectbox>div>div>div {
        border-radius: 20px;
        border: 1px solid var(--primary-light);
    }
    
    /* Badges de statut */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-normal {
        background-color: #E1F5FE;
        color: #0288D1;
    }
    
    .status-attention {
        background-color: #FFF8E1;
        color: #FFA000;
    }
    
    .status-urgent {
        background-color: #FFEBEE;
        color: #D32F2F;
    }
    
    .status-good {
        background-color: #E8F5E9;
        color: #388E3C;
    }
    
    /* Animation de chargement */
    .stProgress > div > div > div > div {
        background-color: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# ----- Fonction pour animation de chargement -----
def loading_animation():
    with st.spinner("Analyse des données médicales en cours..."):
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        st.success("Analyse terminée!")
        time.sleep(0.5)
        st.empty()

# ----- Fonction pour créer une carte métrique médicale -----
def medical_metric_card(title, value, icon, status=None):
    status_html = ""
    if status:
        status_class = {
            "normal": "status-normal",
            "attention": "status-attention",
            "urgent": "status-urgent",
            "good": "status-good"
        }.get(status, "status-normal")
        
        status_text = {
            "normal": "Normal",
            "attention": "À surveiller",
            "urgent": "Urgent",
            "good": "Excellent"
        }.get(status, "Normal")
        
        status_html = f'<span class="status-badge {status_class}">{status_text}</span>'
    
    return f"""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div class="metric-label">{title}</div>
            {status_html}
        </div>
        <div class="metric-value">
            <span style="margin-right: 8px;">{icon}</span>
            {value}
        </div>
    </div>
    """

# ----- Fonction pour créer une carte d'informations patient -----
def patient_info_card(name, age, id, last_visit):
    return f"""
    <div class="patient-card">
        <div style="display: flex; justify-content: space-between;">
            <h3 style="margin: 0; color: var(--primary);">{name}</h3>
            <span class="status-badge status-normal">Dossier #{id}</span>
        </div>
        <div style="margin-top: 10px; display: flex; flex-wrap: wrap;">
            <div style="margin-right: 20px;">
                <span style="opacity: 0.7;">Âge:</span> {age} ans
            </div>
            <div>
                <span style="opacity: 0.7;">Dernière visite:</span> {last_visit}
            </div>
        </div>
    </div>
    """

# ----- Fonction pour créer une carte docteur -----
def doctor_info_card(name, specialty, availability):
    return f"""
    <div class="doctor-card">
        <div style="display: flex; justify-content: space-between;">
            <h3 style="margin: 0; color: var(--secondary);">Dr. {name}</h3>
            <span class="status-badge status-good">Disponible</span>
        </div>
        <div style="margin-top: 10px; display: flex; flex-wrap: wrap;">
            <div style="margin-right: 20px;">
                <span style="opacity: 0.7;">Spécialité:</span> {specialty}
            </div>
            <div>
                <span style="opacity: 0.7;">Prochaine disponibilité:</span> {availability}
            </div>
        </div>
    </div>
    """

# ----- Session state initialization -----
if 'patient_view' not in st.session_state:
    st.session_state.patient_view = True
if 'notifications' not in st.session_state:
    st.session_state.notifications = 3
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        'name': 'Sophie Dupont',
        'age': 34,
        'id': '2023-0456',
        'last_visit': '12/02/2025',
        'appointments': 3
    }

# ----- Définition du menu latéral -----
with st.sidebar:
    image =r"C:\Users\¨PC\Desktop\coursera\345_cancers-femmes-o.jpg"
    st.image(image, use_column_width=True)
    
    # Toggle entre vue patient et médecin
    view_toggle = st.radio(
        "Mode d'affichage",
        ["Vue Patient", "Vue Médecin"],
        horizontal=True,
        index=0 if st.session_state.patient_view else 1
    )
    st.session_state.patient_view = (view_toggle == "Vue Patient")
    
    if st.session_state.patient_view:
        st.markdown(f"### Bonjour, {st.session_state.patient_data['name']}! 👋")
        st.markdown(f"ID Patient: **{st.session_state.patient_data['id']}**")
    else:
        st.markdown("### Bonjour, Dr. Martinez! 👋")
        st.markdown("**👩‍⚕️ Mode Praticien**")
    
    selected = option_menu(
        menu_title="Navigation",
        options=["Accueil", "Dossier Médical", "Résultats d'Analyses", "Rendez-vous", "Messages"],
        icons=["house-heart", "file-earmark-medical", "clipboard2-pulse", "calendar-check", "chat-heart"],
        menu_icon="hospital",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#FFF0F5"},
            "icon": {"color": "#FF69B4", "font-size": "20px"},
            "nav-link": {"color": "#6A0DAD", "font-size": "16px", "text-align": "left", "margin": "0px", "border-radius": "10px"},
            "nav-link-selected": {"background-color": "#FF69B4", "color": "white"},
        }
    )
    
    st.markdown("---")
    
    if st.session_state.patient_view:
        st.markdown("### Votre équipe médicale")
        st.markdown(doctor_info_card("Martinez", "Médecin généraliste", "Aujourd'hui 16:00"), unsafe_allow_html=True)
        st.markdown(doctor_info_card("Richard", "Cardiologue", "22/03/2025"), unsafe_allow_html=True)
    else:
        st.markdown("### Patients du jour")
        st.markdown(patient_info_card("Sophie Dupont", 34, "2023-0456", "12/02/2025"), unsafe_allow_html=True)
        st.markdown(patient_info_card("Marie Laurent", 42, "2022-1287", "02/03/2025"), unsafe_allow_html=True)
        st.markdown(patient_info_card("Camille Noir", 29, "2024-0098", "14/03/2025"), unsafe_allow_html=True)

# ----- Page Accueil -----
# ----- Page Accueil -----
if selected == "Accueil":
    # Header animation avec image de fond subtile
    st.markdown("""
    <div style="text-align:center; padding:20px; background-image: linear-gradient(rgba(255,240,245,0.9), rgba(255,240,245,0.9)), url('https://source.unsplash.com/800x200/?medical,feminine'); background-size:cover; border-radius:15px;">
        <div class="main-header">Bienvenue sur MédiSoin 💗</div>
        <div class="sub-header">Votre partenaire de santé féminine spécialisé en prévention du cancer du col de l'utérus</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Loading animation
    if 'page_loaded' not in st.session_state:
        loading_animation()
        st.session_state.page_loaded = True
    
    # Système intelligent de recommandations personnalisées
    st.markdown("""
    <div style="background-color:white; padding:15px; border-radius:10px; border-left:4px solid #FF69B4; margin-bottom:20px;">
        <h3 style="color:#FF69B4; margin-top:0;">✨ Recommandations intelligentes</h3>
        <p>Notre système a analysé votre dossier et vous suggère les actions suivantes :</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cartes de recommandations intelligentes
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card" style="border-top-color:#FF69B4;">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <span style="background-color:#FFECF3; color:#FF69B4; padding:8px; border-radius:50%; margin-right:10px;">🔔</span>
                <h4 style="margin:0; color:#FF69B4;">Rappel important</h4>
            </div>
            <p>Vos résultats de test HPV indiquent la présence d'une souche à haut risque (HPV-16). Un suivi plus rapproché est recommandé.</p>
            <div style="display:flex; justify-content:space-between; margin-top:15px;">
                <span style="font-size:0.9rem; opacity:0.7;">Priorité: Haute</span>
                <button style="background-color:#FF69B4; color:white; border:none; padding:5px 15px; border-radius:20px; font-size:0.9rem;">Prendre RDV</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card" style="border-top-color:#9370DB; margin-top:20px;">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <span style="background-color:#F5F0FF; color:#9370DB; padding:8px; border-radius:50%; margin-right:10px;">📊</span>
                <h4 style="margin:0; color:#9370DB;">Évaluation personnalisée</h4>
            </div>
            <p>Selon votre profil et vos facteurs de risque, nous recommandons une colposcopie pour examiner plus en détail les anomalies détectées lors de votre dernier test Pap.</p>
            <div style="display:flex; justify-content:space-between; margin-top:15px;">
                <span style="font-size:0.9rem; opacity:0.7;">Action suggérée</span>
                <button style="background-color:#9370DB; color:white; border:none; padding:5px 15px; border-radius:20px; font-size:0.9rem;">En savoir plus</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card" style="border-top-color:#4DB6AC;">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <span style="background-color:#E0F2F1; color:#4DB6AC; padding:8px; border-radius:50%; margin-right:10px;">🍎</span>
                <h4 style="margin:0; color:#4DB6AC;">Conseils santé adaptés</h4>
            </div>
            <p>Pour renforcer votre système immunitaire face au HPV, notre IA recommande les changements de mode de vie suivants basés sur les études récentes :</p>
            <ul style="padding-left:20px;">
                <li>Alimentation riche en antioxydants</li>
                <li>Supplémentation en vitamine D</li>
                <li>Réduction du stress par la méditation</li>
            </ul>
            <div style="text-align:right; margin-top:10px;">
                <button style="background-color:#4DB6AC; color:white; border:none; padding:5px 15px; border-radius:20px; font-size:0.9rem;">Plan personnalisé</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card" style="border-top-color:#FF9800; margin-top:20px;">
            <div style="display:flex; align-items:center; margin-bottom:10px;">
                <span style="background-color:#FFF3E0; color:#FF9800; padding:8px; border-radius:50%; margin-right:10px;">📝</span>
                <h4 style="margin:0; color:#FF9800;">Questionnaire intelligent</h4>
            </div>
            <p>Un nouveau questionnaire personnalisé est disponible pour affiner votre plan de soins. Il prendra moins de 5 minutes à compléter.</p>
            <div style="text-align:right; margin-top:15px;">
                <button style="background-color:#FF9800; color:white; border:none; padding:5px 15px; border-radius:20px; font-size:0.9rem;">Commencer</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Métriques personnalisées - différentes pour patient et médecin
    st.markdown("### 📊 Indicateurs de suivi")
    if st.session_state.patient_view:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(medical_metric_card("Prochaine consultation", "22 Mars", "📅"), unsafe_allow_html=True)
        with col2:
            st.markdown(medical_metric_card("Analyses en attente", "2", "🔬", "attention"), unsafe_allow_html=True)
        with col3:
            st.markdown(medical_metric_card("Messages", "1 nouveau", "💌"), unsafe_allow_html=True)
    
    # Assistant IA pour cancer du col de l'utérus
    st.markdown("### 🤖 Assistant IA - Prévention du cancer du col")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://source.unsplash.com/300x300/?ai,doctor,female", use_column_width=True)
    with col2:
        st.markdown("""
        <div style="background-color:#F8F8FF; padding:15px; border-radius:10px; border:1px solid #E6E6FA;">
            <h4 style="color:#9370DB; margin-top:0;">Dr. IA à votre service</h4>
            <p>Posez-moi vos questions sur le cancer du col de l'utérus, les traitements, la prévention, ou vos symptômes. Je suis là pour vous aider à comprendre votre situation et à prendre des décisions éclairées.</p>
            <div style="background-color:white; padding:10px; border-radius:5px; border:1px solid #E6E6FA;">
                <p><strong>Questions fréquentes :</strong></p>
                <ul style="padding-left:20px; margin-bottom:5px;">
                    <li>Que signifie un résultat ASCUS?</li>
                    <li>Comment le HPV cause-t-il le cancer?</li>
                    <li>Quels sont les effets secondaires d'une colposcopie?</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        user_question = st.text_input("Tapez votre question ici...")
        if st.button("Poser ma question"):
            st.info("Votre question a été transmise à notre système. Une réponse personnalisée apparaîtra ici dans quelques instants.")
    
    # Calendrier interactif intelligent
    st.markdown("### 📆 Votre suivi dans le temps")
    
    # Timeline de suivi
    timeline_data = [
        {"date": "10/01/2025", "event": "Test Pap + HPV", "status": "Réalisé", "result": "ASCUS, HPV+"},
        {"date": "22/03/2025", "event": "Colposcopie", "status": "Planifié", "result": "-"},
        {"date": "05/06/2025", "event": "Résultats & Consultation", "status": "À planifier", "result": "-"},
        {"date": "01/09/2025", "event": "Test Pap de contrôle", "status": "Recommandé", "result": "-"},
        {"date": "15/01/2026", "event": "Bilan annuel", "status": "Recommandé", "result": "-"}
    ]
    
    st.markdown("""
    <div style="overflow-x:auto;">
        <div style="display:flex; min-width:800px; margin-top:20px;">
    """, unsafe_allow_html=True)
    
    for i, item in enumerate(timeline_data):
        status_color = {
            "Réalisé": "#8BC34A",
            "Planifié": "#FF69B4",
            "À planifier": "#FFA000",
            "Recommandé": "#9370DB"
        }.get(item["status"], "#9E9E9E")
        
        result_display = ""
        if item["result"] != "-":
            result_display = f"""<div style="margin-top:5px; font-size:0.8rem; color:#FF69B4;">{item["result"]}</div>"""
            
        st.markdown(f"""
            <div style="flex:1; text-align:center; position:relative; padding-top:25px;">
                <div style="position:absolute; top:0; left:50%; transform:translateX(-50%); width:20px; height:20px; border-radius:50%; background-color:{status_color}; z-index:2;"></div>
                {f'<div style="position:absolute; top:10px; left:0; right:50%; height:2px; background-color:#E0E0E0;"></div>' if i > 0 else ''}
                {f'<div style="position:absolute; top:10px; left:50%; right:0; height:2px; background-color:#E0E0E0;"></div>' if i < len(timeline_data)-1 else ''}
                <div style="background-color:white; border-radius:10px; padding:10px; margin:5px; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-weight:bold; color:{status_color};">{item["date"]}</div>
                    <div>{item["event"]}</div>
                    <div style="margin-top:5px; font-size:0.8rem;">{item["status"]}</div>
                    {result_display}
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Articles et ressources éducatives
    st.markdown("### 📚 Ressources éducatives personnalisées")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color:white; border-radius:10px; overflow:hidden; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
            <div style="height:120px; background-image:url('https://source.unsplash.com/300x150/?women,health'); background-size:cover;"></div>
            <div style="padding:15px;">
                <h4 style="margin-top:0; color:#FF69B4;">Comprendre le test HPV</h4>
                <p style="font-size:0.9rem;">Un guide complet sur les tests HPV, leur interprétation et leur importance dans la prévention du cancer.</p>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.8rem; color:#9E9E9E;">5 min de lecture</span>
                    <button style="background-color:#FF69B4; color:white; border:none; padding:3px 10px; border-radius:20px; font-size:0.8rem;">Lire</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:white; border-radius:10px; overflow:hidden; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
            <div style="height:120px; background-image:url('https://source.unsplash.com/300x150/?doctor,consultation'); background-size:cover;"></div>
            <div style="padding:15px;">
                <h4 style="margin-top:0; color:#9370DB;">Que se passe-t-il lors d'une colposcopie?</h4>
                <p style="font-size:0.9rem;">Vidéo explicative et guide étape par étape pour vous préparer à cet examen important.</p>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.8rem; color:#9E9E9E;">Vidéo 4:30</span>
                    <button style="background-color:#9370DB; color:white; border:none; padding:3px 10px; border-radius:20px; font-size:0.8rem;">Regarder</button>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    

    # Section spécifique pour le cancer du col de l'utérus
    st.markdown("---")
    st.markdown("### 🎗️ Suivi Cancer du Col de l'Utérus")
    
    if st.session_state.patient_view:
        # Vue patiente
        st.info("Votre médecin vous a envoyé un questionnaire important pour votre suivi. Merci de prendre quelques minutes pour y répondre.")
        
        # Onglets d'information pour patientes
        info_tabs = st.tabs(["Informations", "Facteurs de risque", "Symptômes", "Dépistage"])
        
        with info_tabs[0]:
            st.markdown("""
            #### Qu'est-ce que le cancer du col de l'utérus?
            
            Le cancer du col de l'utérus se développe au niveau du col de l'utérus, qui est la partie inférieure étroite de l'utérus qui s'ouvre sur le vagin. La plupart des cancers du col de l'utérus sont causés par différentes souches du virus du papillome humain (VPH), une infection sexuellement transmissible.
            
            #### Points importants:
            - La vaccination contre le VPH peut prévenir la plupart des cas
            - Le dépistage régulier (test Pap) permet une détection précoce
            - Détecté tôt, ce cancer a un excellent taux de guérison
            """)
            
        with info_tabs[1]:
            st.markdown("""
            #### Facteurs de risque principaux
            
            Les facteurs qui peuvent augmenter votre risque de cancer du col de l'utérus incluent:
            - Infection par le VPH (Human Papillomavirus)
            - Tabagisme
            - Immunodépression
            - Contraception hormonale à long terme
            - Plusieurs grossesses
            - Jeune âge lors de la première grossesse
            - Antécédents familiaux de cancer du col de l'utérus
            """)
            
        with info_tabs[2]:
            st.markdown("""
            #### Symptômes à surveiller
            
            Au stade précoce, le cancer du col de l'utérus ne présente généralement pas de symptômes. Les signes qui peuvent apparaître comprennent:
            
            - Saignements vaginaux anormaux (après les rapports sexuels, entre les règles ou après la ménopause)
            - Pertes vaginales inhabituelles
            - Douleurs pendant les rapports sexuels
            - Douleurs pelviennes
            - Fatigue inexpliquée
            
            **Important**: Si vous présentez ces symptômes, consultez rapidement votre médecin. Ce ne sont pas nécessairement des signes de cancer, mais ils doivent être évalués.
            """)
            
        with info_tabs[3]:
            st.markdown("""
            #### Importance du dépistage régulier
            
            Le dépistage permet de détecter des changements cellulaires précancéreux avant qu'ils ne deviennent cancéreux.
            
            **Recommandations actuelles:**
            - Test Pap tous les 3 ans pour les femmes de 21 à 29 ans
            - Test Pap et test VPH tous les 5 ans pour les femmes de 30 à 65 ans
            
            Votre dernier dépistage date du: **10/01/2025**
            
            Prochain dépistage recommandé: **Janvier 2028**
            """)
            
        # Questionnaire pour patientes
        with st.expander("📋 Questionnaire de suivi - Cancer du col de l'utérus"):
            st.markdown("##### Merci de répondre aux questions suivantes pour améliorer votre suivi médical")
            
            col1, col2 = st.columns(2)
            with col1:
                q1 = st.radio("Avez-vous déjà été vaccinée contre le VPH?", ["Oui", "Non", "Je ne sais pas"])
                q2 = st.radio("Avez-vous remarqué des saignements inhabituels?", ["Oui", "Non"])
                q3 = st.radio("Ressentez-vous des douleurs pelviennes?", ["Régulièrement", "Occasionnellement", "Non"])
                
            with col2:
                q4 = st.radio("Fumez-vous?", ["Oui", "Non", "J'ai arrêté"])
                q5 = st.radio("Votre dernière consultation gynécologique date de:", ["Moins de 1 an", "1-3 ans", "Plus de 3 ans"])
                q6 = st.radio("Avez-vous des antécédents familiaux de cancer?", ["Oui", "Non", "Je ne sais pas"])
            
            st.text_area("Autres symptômes ou préoccupations que vous souhaitez signaler:", height=100)
            
            if st.button("Soumettre mes réponses"):
                st.success("Merci d'avoir complété le questionnaire! Vos réponses ont été enregistrées et seront examinées par votre médecin lors de votre prochain rendez-vous.")
    
    else:
        # Vue médecin - section cancer du col
        st.markdown("#### Module de suivi des patientes - Cancer du col de l'utérus")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            ##### Statistiques
            - **Patientes suivies:** 42
            - **Dépistages à programmer:** 8
            - **Rappels envoyés:** 15
            - **Questionnaires complétés:** 26
            """)
            
            st.markdown("##### Actions rapides")
            st.button("Envoyer rappels de dépistage")
            st.button("Générer rapport mensuel")
            
        with col2:
            # Tableau de bord médecin - vue des patientes à risque
            risk_data = pd.DataFrame({
                'Patiente': ['Sophie Dupont', 'Marie Laurent', 'Julia Martin', 'Amina Benali', 'Lucie Moreau'],
                'Âge': [34, 42, 51, 29, 47],
                'Dernier dépistage': ['10/01/2025', '05/09/2024', '22/03/2023', '17/02/2025', '30/11/2024'],
                'Facteurs de risque': ['VPH+', 'Tabagisme', 'Antécédents familiaux', 'Aucun', 'Immunodépression'],
                'Niveau de risque': ['Moyen', 'Élevé', 'Élevé', 'Faible', 'Moyen']
            })
            
            st.markdown("##### Patientes à surveiller")
            
            # Formater le dataframe avec des couleurs selon le niveau de risque
            def highlight_risk(val):
                if val == 'Élevé':
                    return 'background-color: #FFEBEE; color: #D32F2F'
                elif val == 'Moyen':
                    return 'background-color: #FFF8E1; color: #FFA000'
                else:
                    return 'background-color: #E8F5E9; color: #388E3C'
            
            st.dataframe(risk_data.style.applymap(highlight_risk, subset=['Niveau de risque']), use_container_width=True)
        
        # Questionnaire à utiliser avec la patiente
        with st.expander("📝 Questionnaire d'évaluation - Cancer du col de l'utérus"):
            st.markdown("### Questions à poser à votre patiente")
            
            questions = [
                "1. Avez-vous remarqué des saignements inhabituels (après les rapports sexuels, entre les règles)?",
                "2. Ressentez-vous des douleurs pelviennes ou pendant les rapports sexuels?",
                "3. Avez-vous observé des pertes vaginales inhabituelles (quantité, couleur, odeur)?",
                "4. Quelle est la date de votre dernier test Pap / frottis?",
                "5. Avez-vous été vaccinée contre le HPV? Si oui, à quel âge?",
                "6. Y a-t-il des antécédents de cancer du col de l'utérus dans votre famille?",
                "7. Fumez-vous ou avez-vous fumé par le passé?",
                "8. Depuis combien de temps utilisez-vous votre contraception actuelle?",
                "9. Avez-vous eu plusieurs partenaires sexuels?",
                "10. Avez-vous d'autres préoccupations concernant votre santé gynécologique?"
            ]
            
            for q in questions:
                st.markdown(f"**{q}**")
                st.text_area(f"Notes pour {q}", "", key=f"note_{q}", height=60)
            
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Recommandation:", [
                    "Sélectionner une action",
                    "Programmer un test Pap",
                    "Test HPV",
                    "Colposcopie",
                    "Biopsie",
                    "Référer à un spécialiste",
                    "Vaccination HPV",
                    "Suivi dans 6 mois",
                    "Suivi annuel standard"
                ])
            
            with col2:
                st.date_input("Date du prochain rendez-vous:")
            
            if st.button("Enregistrer consultation"):
                st.success("Consultation enregistrée avec succès dans le dossier de la patiente.")

# ----- Page Dossier Médical -----
elif selected == "Dossier Médical":
    st.markdown('<div class="main-header">Dossier Médical</div>', unsafe_allow_html=True)
    
    if st.session_state.patient_view:
        st.markdown('<div class="sub-header">Votre historique et informations médicales</div>', unsafe_allow_html=True)
        
        # Informations générales
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🧬 Informations générales")
            st.markdown("""
            **Nom:** Sophie Dupont  
            **Date de naissance:** 12/06/1991  
            **Groupe sanguin:** A+  
            **Allergies:** Aucune connue  
            **Médicaments actuels:** Contraceptif oral  
            """)
        
        with col2:
            st.markdown("### 📊 Facteurs de risque")
            st.markdown("""
            **VPH:** Positif (souche 16)  
            **Tabagisme:** Non  
            **Contraception hormonale:** Oui (5 ans)  
            **Antécédents familiaux:** Cancer du sein (mère)  
            """)
        
        # Historique médical
        st.markdown("### 📜 Historique médical")
        history_data = pd.DataFrame({
            'Date': ['10/01/2025', '15/09/2024', '22/03/2024', '05/01/2024', '12/06/2023'],
            'Type': ['Dépistage', 'Consultation', 'Vaccination', 'Test sanguin', 'Examen gynécologique'],
            'Détails': ['Test Pap + Test HPV', 'Suivi contraception', 'HPV (3ème dose)', 'Bilan hormonal', 'Examen annuel'],
            'Résultats': ['En attente', 'Normal', 'Complété', 'Normal', 'Normal']
        })
        
        st.dataframe(history_data, use_container_width=True)
        
    else:
        # Vue médecin
        st.markdown('<div class="sub-header">Dossier médical de la patiente</div>', unsafe_allow_html=True)
        
        # Sélection de patiente
        patient_selected = st.selectbox(
            "Sélectionner une patiente:",
            ["Sophie Dupont", "Marie Laurent", "Julia Martin", "Amina Benali", "Lucie Moreau"]
        )
        
        # Informations spécifiques cancer du col
        st.markdown("### 🎗️ Suivi Cancer du Col - Sophie Dupont")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Résultats derniers tests")
            st.markdown("""
            **Test Pap (10/01/2025):** ASCUS (cellules squameuses atypiques de signification indéterminée)  
            **Test HPV:** Positif (souche 16 - haut risque)  
            **Colposcopie:** Non réalisée  
            **Biopsie:** Non réalisée  
            """)
            
            st.markdown("#### Facteurs de risque")
            risk_factors = {
                "VPH à haut risque": True,
                "Tabagisme": False,
                "Immunodépression": False,
                "Contraception hormonale >5 ans": True,
                "Partenaires multiples": False,
                "Première grossesse précoce": False,
                "Antécédents familiaux": True
            }
            
            for factor, present in risk_factors.items():
                icon = "✅" if present else "❌"
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. FEATURE EXTRACTION (The "Brain" of the AI) ---
# This function turns a text URL into numbers the AI can understand.
def extract_features(url):
    features = []
    
    # Feature 1: Length of URL (Phishing URLs are often very long)
    features.append(len(url))
    
    # Feature 2: Count of dots (More than 3 dots is suspicious)
    features.append(url.count('.'))
    
    # Feature 3: Presence of '@' symbol (Common in phishing)
    features.append(1 if '@' in url else 0)
    
    # Feature 4: Count of sensitive words (e.g., 'login', 'secure', 'bank')
    suspicious_words = ['login', 'secure', 'account', 'update', 'verify', 'bank']
    word_count = sum(1 for word in suspicious_words if word in url.lower())
    features.append(word_count)
    
    # Feature 5: Is it using HTTPS? (1 for yes, 0 for no)
    features.append(1 if 'https' in url.lower() else 0)
    
    return features

# --- 2. LOAD DATA & TRAIN MODEL ---
@st.cache_resource # Caches the model so we don't retrain on every click
def train_model():
    # FOR THE HACKATHON:
    # Ideally, you would load the Kaggle CSV here: df = pd.read_csv('phishing_data.csv')
    # For this Demo, we will generate synthetic data so the code RUNS INSTANTLY.
    
    # Synthetic 'Safe' URLs
    safe_urls = ['google.com', 'youtube.com', 'facebook.com', 'wikipedia.org', 'amazon.com']
    # Synthetic 'Phishing' URLs (mimicking patterns)
    phish_urls = ['secure-login-updates.com', 'bank-verify-account.net', 'account-update@security.com',
                  'verify.login.secure.com', 'wp-admin-login.xyz']
    
    data = []
    labels = []
    
    # Process Safe URLs (Label = 0)
    for url in safe_urls:
        data.append(extract_features(url))
        labels.append(0) 
        
    # Process Phishing URLs (Label = 1)
    for url in phish_urls:
        data.append(extract_features(url))
        labels.append(1)
        
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['Length', 'Dots', 'Has_At', 'Sus_Words', 'HTTPS'])
    
    # Train the Random Forest Model
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(df, labels)
    
    return clf

# --- 3. THE STREAMLIT WEB INTERFACE ---
def main():
    # Page Config
    st.set_page_config(page_title="CyberGuard AI", page_icon="🛡️")
    
    # Header
    st.title("🛡️ CyberGuard: AI Phishing Predictor")
    st.markdown("""
    **Hackathon Challenge:** Harnessing AI for Predictive Cyber Defense.
    This tool analyzes URL patterns to **predict** if a link is safe or malicious before you click.
    """)
    
    # Sidebar for About
    st.sidebar.header("About the Project")
    st.sidebar.info("This tool uses a Random Forest Classifier to detect anomalies in URL structures, identifying potential zero-day phishing attacks.")
    
    # Load Model
    model = train_model()
    
    # Input Section
    url_input = st.text_input("Enter a URL to analyze:", placeholder="e.g., http://secure-login-bank-update.com")
    
    if st.button("Analyze Risk"):
        if url_input:
            # 1. Extract Features
            input_features = extract_features(url_input)
            
            # 2. Predict
            prediction = model.predict([input_features])[0]
            probability = model.predict_proba([input_features])[0][1] # Probability of being phishing
            
            # 3. Display Results
            st.divider()
            if prediction == 1 or probability > 0.6: # High Risk
                st.error("🚨 DANGER DETECTED: High Phishing Risk!")
                st.metric(label="Phishing Probability", value=f"{probability*100:.1f}%")
                st.write("This URL contains suspicious patterns commonly found in cyber attacks.")
            else: # Safe
                st.success("✅ SAFE: No Threats Detected")
                st.metric(label="Safety Score", value=f"{(1-probability)*100:.1f}%")
                st.write("This URL structure appears legitimate.")
                
            # Show the "Why" (Explainability)
            with st.expander("See Analysis Details (Why did AI think this?)"):
                st.write(f"**URL Length:** {input_features[0]} chars")
                st.write(f"**Suspicious Keywords Found:** {input_features[3]}")
                st.write(f"**Unusual Dots Count:** {input_features[1]}")
        else:
            st.warning("Please enter a URL first.")

if __name__ == "__main__":
    main()
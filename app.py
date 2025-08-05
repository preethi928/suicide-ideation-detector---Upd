import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Page config
st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .chat-container {
            background-color: #f2f2f2;
            padding: 20px;
            border-radius: 15px;
        }
        .user-msg {
            color: #333;
            background-color: #d9fdd3;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .bot-msg {
            color: #fff;
            background-color: #4a90e2;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        a {
            color: #fff;
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Real-Time Mental Health Support Chatbot")

st.markdown("💬 *Type how you're feeling. This bot offers support but is not a substitute for professional help.*")

# Chat session history
if "history" not in st.session_state:
    st.session_state.history = []

# Input form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("🗣️ How are you feeling today?")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    # Prediction
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)[0]

    # Response based on prediction
    if prediction == 1:
        response = (
            "😢 It seems you might be feeling overwhelmed or distressed.\n\n"
            "🧷 You're **not alone**. Please consider talking to someone you trust.\n\n"
            "📞 **Hotlines & Help:**\n"
            "- [Lifeline Chat (USA)](https://988lifeline.org/chat/)\n"
            "- [Bahrain National Hotline: 999](tel:999)\n"
            "- [BetterHelp Online Therapy](https://www.betterhelp.com/)\n\n"
            "💙 You matter. Your story matters."
        )
    else:
        response = (
            "😊 I'm glad you reached out! Keep taking care of yourself.\n\n"
            "🧘‍♀️ A small walk, deep breath, or journaling may help today.\n\n"
            "📚 [Explore mental health tips](https://www.mentalhealth.gov)"
        )

    # Save history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# Display conversation
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, msg in st.session_state.history:
    css_class = "user-msg" if sender == "You" else "bot-msg"
    st.markdown(f'<div class="{css_class}"><strong>{sender}:</strong><br>{msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

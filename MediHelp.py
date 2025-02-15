import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from feedback_handler import FeedbackHandler
from responses import MEDICAL_RESPONSES
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Download necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the translator
translator_to_english = GoogleTranslator(source='auto', target='en')
translator_to_hinglish = GoogleTranslator(source='en', target='hi')

# Initialize feedback handler
feedback_handler = FeedbackHandler()

# Initialize medical model
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Preprocess user input
def preprocess_input(user_input):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(user_input)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Define healthcare-specific response logic
def healthcare_chatbot(user_input, chat_history):
    if not user_input.strip():
        return "Please enter a valid question.", chat_history
    
    # Translate Hinglish to English
    try:
        translated_input = translator_to_english.translate(user_input)
    except Exception as e:
        logger.error(f"Translation error: {e}")
        translated_input = user_input
    
    user_input = preprocess_input(translated_input).lower()
    
    # Append user input to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    # Use pre-trained responses if available
    response_text = MEDICAL_RESPONSES.get(user_input)
    
    if response_text:
        # If an exact match is found, use it immediately
        chat_history.append({"role": "assistant", "content": response_text})
        return response_text, chat_history
    else:
        # Check for variations in user input
        for key in MEDICAL_RESPONSES.keys():
            if user_input in key.lower() or key.lower() in user_input or user_input in key.lower().split():
                response_text = MEDICAL_RESPONSES[key]
                break
        else:
            response_text = None
    
    if not response_text:
        # Generate response using the medical model
        try:
            response_text = generator(user_input, max_length=150, num_return_sequences=1)[0]['generated_text']
        except Exception as e:
            logger.error(f"Model generation error: {e}")
            response_text = "I am still learning. Please check with a doctor for accurate information."
    
    # Append model response to chat history
    chat_history.append({"role": "assistant", "content": response_text})
    return response_text, chat_history

def main():
    st.markdown("""
        <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: auto;
        }
        .title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
        .chat-history {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .user-message {
            color: #007bff;
            margin-bottom: 10px;
        }
        .assistant-message {
            color: #28a745;
            margin-bottom: 10px;
        }
        .input-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .input-container input {
            flex: 1;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #ccc;
            margin-right: 10px;
        }
        .input-container button {
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
            background-color: #007bff;
            color: #fff;
            cursor: pointer;
        }
        .input-container button:hover {
            background-color: #0056b3;
        }
        .prompt {
            font-size: 1.2em;
            color: #333;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">MediHelp AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="prompt">How can I help you today?</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        role = "User" if message["role"] == "user" else "Healthcare Assistant"
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(f'<div class="{role_class}"><strong>{role}:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get user input
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.text_input("You: ", "")
    if st.button("Send"):
        response, st.session_state.chat_history = healthcare_chatbot(user_input, st.session_state.chat_history)
        st.markdown(f'<div class="assistant-message"><strong>Healthcare Assistant:</strong> {response}</div>', unsafe_allow_html=True)
        
        # Collect feedback
        is_helpful = st.radio("Was this response helpful?", ("Yes", "No"))
        user_comment = st.text_area("Additional comments:")
        if st.button("Submit Feedback"):
            feedback_handler.add_feedback(user_input, response, is_helpful == "Yes", user_comment)
            st.success("Thank you for your feedback!")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
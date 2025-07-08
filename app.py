from embeddings import  genarate_response
from langchain.memory import ConversationBufferMemory
from start_speaking_test import IELTSSpeakingTest
import streamlit as st 
import time



# Streamlit UI     

gradient_text_html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;900&display=swap');

.title {
  font-family: 'Poppins', sans-serif;
  font-weight: 900;
  font-size: 4em;
  background: linear-gradient(90deg, #ff6a00, #ee0979);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
  margin: 0;
  padding: 20px 0;
  text-align: center;
}
</style>

<div class="title">IELTS PRO</div>
"""

# Insert for title

st.markdown(gradient_text_html, unsafe_allow_html=True)

# Styling for the speaking button
st.markdown("""<style>
.custom-button-style button {
background-color: #ff416c;
color: white;
border: none;
padding: 12px 30px;
border-radius: 10px;
font-size: 18px;
font-weight: bold;
cursor: pointer;
} </style>
""", unsafe_allow_html=True)


# Initialize chat history 
if "messages" not in st.session_state:
    st.session_state.messages=[]

# Initialize the session for speaking test
if "Speaking_mode" not in st.session_state:
    st.session_state.Speaking_mode = False
    
if "langchain_memory" not in st.session_state:
    st.session_state.langchain_memory = ConversationBufferMemory(memory_key="chat_history")

# Initialize the session for chat_mode
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = True
    


# Speaking test button

col1, col2, col3 = st.columns(3)
with col2:
    with st.container():
        if st.button("🎙️ Start Speaking Test", key="start"):
            st.session_state.chat_mode = False    # Switch to Speaking test view
            st.session_state.Speaking_mode = True

# If chat_mode is active, show chat interface
if st.session_state.chat_mode:
    
    # show message history
    for msg in st.session_state.get("messages", []):
        st.chat_message(msg["role"]).markdown(msg["content"])      
    
    # React to user input
    if prompt := st.chat_input("Ask IELTS"):
    
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
    
        # Add user message to chat history 
        st.session_state.messages.append({"role":"user", "content": prompt})

    
   
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
            
                response_placeholder = st.empty()
                full_response = ""
                for sentence in genarate_response(prompt):
                    full_response += sentence + " "
                    response_placeholder.write(full_response)
                    time.sleep(0.5)
                               
            # Add assistant reposne to chat history
            st.session_state.messages.append({"role":"assistant", "content": full_response})
            
            # Save context to langchain memory
            st.session_state.langchain_memory.save_context(
                {"input": prompt},
                {"output": full_response})
        
    
    
                
else:
    
    # Hide chat, how Speaking test interface
    IELTSSpeakingTest().start_test()
    
    # Show button to go back to chat
    if st.button("🔙 Back to Chat"):
        st.session_state.chat_mode = True
        st.session_state.Speaking_mode = False
    
import os
import pickle
import secrets
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room

# Create a Flask app and generate a secret key for it
app = Flask(__name__, static_url_path='', static_folder='static')
app.config['SECRET_KEY'] = secrets.token_hex(32)

# Initialize a SocketIO instance for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*")

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Import necessary modules
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Load the Faiss index from a pickle file
with open("words.pkl", 'rb') as f:
    faiss_index = pickle.load(f)

# Create a ChatOpenAI instance with specified settings
chat = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo')

# Initialize variables for message history
message_history = []

# Create a dictionary to store room-specific message history
room_message_history = {}

# Create an initial system message
messages = [
    SystemMessage(content="""
    You are a translation assistant that translates Turkish words and sentences into English. 
    Please only perform translations, 
    and let me know that you can only do translations for all other questions.
    """)
]

# Define a function for making predictions based on user input
def predict(input: str):
    global message_history
    # Use Faiss index for similarity search
    docs = faiss_index.similarity_search(input, K=1)
    main_content = input + "\n\n"
    for doc in docs:
        main_content += doc.page_content + "\n\n"

    # Append the user's input to the message history
    message_history.append({"role": "user", "content": f"{input}"})

    # Add the user's input as a human message to the messages list
    messages.append(HumanMessage(content=input))

    # Generate an AI response based on the user's input
    ai_response = chat(messages).content

    # Remove the user's input and add AI's response to the messages list
    messages.pop()
    messages.append(HumanMessage(content=main_content))
    messages.append(AIMessage(content=ai_response))

    # Append the AI's response to the message history
    message_history.append({"role": "assistant", "content": f"{ai_response}"})

    # Create a response as a list of user and assistant message pairs
    response = [(message_history[i]["content"], message_history[i + 1]["content"]) for i in
                range(0, len(message_history) - 1, 2)]
    return response

# Define a route for the root endpoint (GET and POST methods)
@app.route("/", methods=["GET", "POST"])
def api_root():
    if request.method == "GET":
        return {"message": "Welcome to the Chatbot API"}
    elif request.method == "POST":
        input_data = request.json
        input = str(input_data.get("input"))
        if input:
            response_data = predict(input)
            return jsonify(response_data)
        else:
            return {"error": "Missing 'input' in the request body"}

# Create a dictionary to store user rooms
user_rooms = {}

# Handle client socket connection
@socketio.on('connect')
def handle_connect():
    session_id = secrets.token_hex(16)
    join_room(session_id)
    emit('join', {'session_id': session_id})

# Handle client socket disconnection
@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.args.get('session_id')
    leave_room(session_id)
    emit('leave', {'session_id': session_id})

# Handle incoming messages from the client via Socket.IO
@socketio.on('message_from_client')
def handle_message(message):
    session_id = message.get('session_id')
    user_input = message.get('content')

    if not isinstance(user_input, str):
        return

    response_data = predict(user_input)
    emit('update_chat', response_data, room=session_id)

# Run the Flask app with the Socket.IO server
if __name__ == "__main__":
    socketio.run(app, host='127.0.0.1', port=8080, debug=False)

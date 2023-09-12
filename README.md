# Langchain-Python-Custom-Chatbot


## Overview

This repository contains a custom chatbot built using the Langchain Python library. 
The chatbot can translate Turkish words and sentences into English and engage in real-time conversations with users via a web interface.
 It utilizes the GPT-3.5 Turbo model from OpenAI for natural language understanding and generation.

## Features

- Real-time chat interface using Flask and Socket.IO.
- Translation of Turkish words and sentences to English.
- Faiss index for similarity search of translated words.
- Interactive conversation with the chatbot.

## Prerequisites

Before running the chatbot, make sure you have the following installed:

- Python 3.6 or higher
- Flask
- Flask-SocketIO
- OpenAI Python library
- Faiss (for the similarity search)
-Langchain

You'll also need to obtain an API key from OpenAI and replace `"your-api-key"` in the code with your actual API key.


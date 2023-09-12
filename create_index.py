import pickle
import json
from pathlib import Path
import faiss
import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Load data from JSON file
with open("words.json", "r") as json_file:
    json_data = json.load(json_file)

# Extract text, turkish_words, and english_words fields from JSON data
english_words = [item["english_words"] for item in json_data]
turkish_words = [item["turkish_words"] for item in json_data]

# Combine turkish_words and english_words into a dictionary
word_translation_dict = dict(zip(turkish_words, english_words))

# Create a text splitter to divide text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
documents = []
metadata_list = []

# Split and process each Turkish-English word pair
for turkish_word, english_word in zip(turkish_words, english_words):
    text = f"Turkish Word: {turkish_word}\nEnglish Word: {english_word}"
    splits = text_splitter.split_text(text)
    documents.extend(splits)
    metadata_list.extend([{"turkish_word": turkish_word, "english_word": english_word}] * len(splits))

# Identify and remove documents longer than 1600 characters
long_documents = [i for i, d in enumerate(documents) if len(d) > 1600]

for i in sorted(long_documents, reverse=True):
    print('Removing document due to size', f'Size: {len(documents[i])} Document: {documents[i]}')
    del documents[i]
    del metadata_list[i]

# Create a vector store from the documents
store = FAISS.from_texts(documents, OpenAIEmbeddings(), metadatas=metadata_list)

# Replace the existing index with the new vector store
faiss.write_index(store.index, "docs.index")

# Save the vector store to a pickle file
with open("words.pkl", "wb") as f:
    pickle.dump(store, f)

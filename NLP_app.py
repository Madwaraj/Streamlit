import streamlit as st
import pandas as pd
import numpy as np

import spacy
from textblob import TextBlob
from gensim.summarization import summarize

import nltk
nltk.download('punkt')

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx, Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document, 3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

def text_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	tokens = [token.text for token in docx]
	allData = [(' "Tokens":{},\n "Lemma":{} \n'.format(token.text, token.lemma_)) for token in docx]
	return allData

def entity_analyzer(my_text):
	nlp = spacy.load('en_core_web_sm')
	docx = nlp(my_text)
	entities = [(entity.text, entity.label_) for entity in docx.ents]
	return entities

def main():
	st.title("NLP with streamlit")
	st.subheader("Interactive NLP")

	#Tokenization
	if st.checkbox("Show tokens and lemmas"):
		st.subheader("Tokenize your text")
		message = st.text_area("Enter your text", "Type here")	
		if st.button("Analyze"):
			nlp_results = text_analyzer(message)
			st.json(nlp_results)

	# Named Entity
	if st.checkbox("Show Named entities"):
		st.subheader("Get entitites from your text")
		message = st.text_area("Enter your text", "Type here")	
		if st.button("Extract"):
			nlp_results = entity_analyzer(message)
			st.json(nlp_results)

	# Sentiment analysis
	if st.checkbox("Show sentiment analysis"):
		st.subheader("Get sentiment from your text")
		message = st.text_area("Enter your text", "Type here")	
		if st.button("Analyze sentiment"):
			blob = TextBlob(message)
			nlp_results = blob.sentiment	
			st.success(nlp_results)

	# Text summarization
	if st.checkbox("Show text summarization"):
		st.subheader("Summarize your text")
		message = st.text_area("Enter your text", "Type here")
		summary_options = st.selectbox("Select your summarizer", ("gensim", "sumy"))	
		if st.button("Summarize text"):
			if summary_options == "gensim":
				st.text("Using gensim")
				summary_result = summarize(message)
			elif summary_options == "sumy":
				st.text("Using sumy")
				summary_result = sumy_summarizer(message)
			else:
				st.warning("Using default summarizer")
				st.text("Using Gensim")
				summary_result = summarize(message)
			st.success(summary_result)
	
	st.sidebar.subheader("About App")
	st.sidebar.text("NLP App with streamlit")
	st.sidebar.info("Practicing with streamlit")

if __name__ == '__main__':
	main()

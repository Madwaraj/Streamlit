import streamlit as st
import urllib
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead 

def load_bad_words() -> list:
	res_list = []

	file = urllib.request.urlopen("https://raw.githubusercontent.com/RobertJGabriel/Google-profanity-words/master/list.txt")
	for line in file:
		dline = line.decode("utf-8")
		res_list.append(dline.split("\n")[0])
	return res_list

BAD_WORDS = load_bad_words()

def filter_bad_words(text):
	explicit = False

	res_text = text.lower()
	for word in BAD_WORDS():
		if word in res_text:
			res_text = res_text.replace(word, word[0] + "*"*len(word[1:]))

		output_text = ""
		for oword, rword in zip(text.split(" "), res_text.split(" ")):
			if oword.lower() == rword:
				output_text += oword + " "
			else:
				output_text += rword + " " 
	return output_text

@st.cache(allow_output_mutation = True, suppress_st_warning = True)
def load_model():
	return pipeline("text-generation", model = "e-tony/gpt2-rnm")

model = load_model()

textbox = st.text_area("Start your story"," ", height = 200, max_chars = 1000)

slider = st.slider("Max story length (in characters)", 50, 200)

button = st.button("Generate")

if button:
	unfiltered_text = model(textbox, do_sample=True, max_length=slider, top_k=20, top_p=0.95, num_returned_sequence=1)[0]['generated_text']
	output_text = filter_bad_words(unfiltered_text)
	for i, line in enumerate(output_text.split("\n")):
		if ":" in line:
			speaker, speech = line.split(':')
			st.markdown(f'__{speaker}__: {speech}')
		else:
			st.markdown(line)




from transformers import pipeline
from transformers import PegasusForConditionalGeneration,PegasusTokenizer


model_name = "google/pegasus-xsum"

# Load the Pegasus tokenizer and model
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Input paragraph
para = input("Enter your paragraph: ")

print("Original document size:", len(para))

# Tokenize the input paragraph
tokens = pegasus_tokenizer(para, truncation=True, padding="longest", return_tensors="pt")

# Generate the summary
generated_summary = pegasus_model.generate(**tokens)

# Decode the generated summary
decoded_summary = pegasus_tokenizer.decode(generated_summary[0], skip_special_tokens=True)

print("Decoded Summary:", decoded_summary)

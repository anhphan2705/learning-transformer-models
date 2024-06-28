from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
import torch

# A sequence to encode
sequence = "I'm testing out my knowledge on Hugging Face Transformer encoder"
print(f"This is the sequence that will be translated: {sequence}\n")

# Setting up the environment
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# model = AutoModel.from_pretrained("")
bert_config = BertConfig()
bert_model = BertModel(bert_config)  # This will randomly generate gibberish
bert_model = BertModel.from_pretrained("bert-base-cased")

# Print out the result in each steps of the encoder
tokens = tokenizer.tokenize(sequence)
print(f"These are the tokens: {tokens}\n")
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"These are the token ids: {token_ids}\n")

# Decoded string
decoded_string = tokenizer.decode(token_ids)
print(f"The decoded string from the token ids: {decoded_string}\n")

# Get the tokenized sequence
encoded_sequence = tokenizer(sequence)      # tokenizer(sequence, return_tensors="pt") to be able to input to the model
print(f"This is the encoded sequence: {encoded_sequence}\n")

# Turn the encoded sequence into tensors to input to the model
model_inputs = torch.tensor([token_ids])    # Model expects more than 1 setence, hence add another layer
print(model_inputs.shape)

# Get the output from the pretrained model
outputs = bert_model(model_inputs)
print(f"Pre-Softmax outputs {outputs}\n")
# outputs = torch.nn.functional.softmax(outputs)
# print(f"Post-Softmax outputs {outputs}\n")

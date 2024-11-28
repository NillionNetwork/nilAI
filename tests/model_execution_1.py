import time

from onnxruntime import InferenceSession
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Define the model directory and ONNX export location
model_name = "meta-llama/Llama-3.2-1B-Instruct"
onnx_export_dir = "./onnx_model"

# Export the model
model = ORTModelForCausalLM.from_pretrained(model_name, from_transformers=True)
model.save_pretrained(onnx_export_dir)

# Save the tokenizer for later use
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(onnx_export_dir)


# Load the ONNX model and tokenizer
onnx_model_path = "./onnx_model/model.onnx"
tokenizer = AutoTokenizer.from_pretrained("./onnx_model")

# Create an ONNX Runtime session
session = InferenceSession(onnx_model_path)

# Input messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"},
]

# Prepare input text
input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

# Tokenize input text
inputs = tokenizer(input_text, return_tensors="pt")
print("START:")
# Run inference
start = time.time()
onnx_inputs = {session.get_inputs()[0].name: inputs["input_ids"].numpy()}
onnx_output = session.run(None, onnx_inputs)
end = time.time()

# Decode the output
output_text = tokenizer.decode(onnx_output[0][0], skip_special_tokens=True)

print(output_text)
print(f"Time taken: {end - start} seconds")

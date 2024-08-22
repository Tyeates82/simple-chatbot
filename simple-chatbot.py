from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Utilizing Facebook Blenderbot for processing
model_name = 'facebook/blenderbot-400M-distill'

# Initialize model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize empty list for conversation history
conversation_history = []

if __name__ == "__main__":
    # Implementing loop for repetition
    while True:
        # Converting conversation history list into the required format
        history_string = '\n'.join(conversation_history)

        # Sample text input
        input_text = input('> ')

        # Processing and returning input
        inputs = tokenizer.encode_plus(history_string, input_text, return_tensors='pt')

        # Generate response using the model
        outputs = model.generate(**inputs)

        # Decoding the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        print(response)

        # Updating conversation history
        conversation_history.append(input_text)
        conversation_history.append(response)

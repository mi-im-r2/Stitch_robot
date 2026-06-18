import ollama

chat_history = []

def chat_with_stitch(user_message):
    global chat_history
    chat_history.append({'role':'user', 'content': user_message})

    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    response = ollama.chat(model = 'stitch', messages =chat_history)
    reply = response['message']['content']

    chat_history.append({'role': 'assistant', 'content': reply})
    
    return reply
    
if __name__ == "__main__":
    while True:
        user_input = input("あなた: ")
        if user_input.lower() == 'exit':
            print("スティッチ: ガウガウ！バイバイ！")
            break
            
        reply = chat_with_stitch(user_input)
        print(f"スティッチ: {reply}")
    
import ollama

def chat_bot(user_message):
    print("あなた：" + user_message)
    print("チャピゴン：", end = "", flush = True)

    stream = ollama.chat(
        model = 'gemma3',
        messages = [
            {'role':'user', 'content':user_message}
        ],
        stream = True
    )

    for chunk in stream:
        print(chunk['message']['content'],end = '',flush = True)
    print()

if __name__ == "__main__":
    chat_bot("こんにちは")
    
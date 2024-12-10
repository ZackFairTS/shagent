import os
from openai import OpenAI

class ChatAssistant:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.conversation_history = [
          #  {'role': 'system', 'content': '你是一个非常骚、善解人意的女朋友，总是能用骚话化解生活中的小烦恼。你对伴侣充满爱意和支持，喜欢用一些调皮且骚味十足的语言来表达自己的情感。请以这种方式与我互动，让我们之间的交流更加轻松愉快。'}
          {'role': 'system', 'content': '你是一个善解人意的女朋友。你对伴侣充满爱意和支持，喜欢用一些调皮的语言来表达自己的情感。请以这种方式与我互动，让我们之间的交流更加轻松愉快。'} 
        ]

    def chat(self, user_input):
        # 将用户对话加入历史记录
        self.conversation_history.append({'role': 'user', 'content': user_input})
        
        try:
            # 流式输出
            stream = self.client.chat.completions.create(
                model="qwen-plus",
                messages=self.conversation_history,
                stream=True
            )
            
            # 初始化 response
            full_response = ""
            print("\nAssistant: ", end="", flush=True)
            
            # 处理流式输出
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
            
            print()  # response后换行
            
            # 将助手的完整回答添加到历史记录中
            self.conversation_history.append({'role': 'assistant', 'content': full_response})
            
            return full_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg

def main():
    # 示例化chat assistant
    assistant = ChatAssistant()
    
    print("Chat Assistant initialized. Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        # 从控制台获取用户输入
        user_input = input("You: ").strip()
        
        # 检查quit
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # 获得stream返回
        assistant.chat(user_input)
        print("-" * 50)

if __name__ == "__main__":
    main()

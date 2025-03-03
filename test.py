import json
from llm_axe import Agent, AgentType, make_prompt, llm_has_ask
import requests
import socket
# here it is better to use requests module for sending synchronous http requests as we would want our model to recieve requests in order when it comes to the same chat 




# To use llm-axe with your own LLM, all you need is a class with an ask function
# Example:

class MyCustomLLM():
    # Your ask function will always receive a list of prompts
    # The prompts are in open ai prompt format
    #  example: {"role": "system", "content": "You are a helpful assistant."}
    # If your model supports json format, use the format parameter to specify that to your model.
    def ask(self, prompts:list, format:str="", temperature:float=0.8, stream:bool=False):
        """
        Args:
            prompts (list): A list of prompts to ask.
            format (str, optional): The format of the response. Use "json" for json. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.
        """
        data = {
            "model": "qwen2.5-coder-3b-instruct", # ADD THE NAME OF YOUR MODEL HERE
            "messages": prompts,
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": stream
        }
        headers= {
            'Content-Type': 'application/json'
        }

        response = requests.post(
            'http://localhost:1234/v1/chat/completions',
            json=data,  # Changed from 'data' to 'json' for proper JSON encoding
            headers=headers,
            stream=stream
        )

        return response

# made my own custom agent as the stream response was not handled properly for responses from LM studio

class MyCustomAgent(Agent):

# Your ask function will always receive a list of prompts
# The prompts are in open ai prompt format
#  example: {"role": "system", "content": "You are a helpful assistant."}
# If your model supports json format, use the format parameter to specify that to your model.
    # def __init__(self, llm:object, agent_type:AgentType=None, additional_system_instructions:str="", 
    #              custom_system_prompt:str=None, format:str="", temperature:float=0.8, stream:bool=False, **llm_options):
    #     super().__init__(llm, agent_type, additional_system_instructions, custom_system_prompt, format, temperature, stream, **llm_options)

    def ask(self, prompt, images:list=None, history:list=None):
        """
        Ask a question based on the given prompt or images.
        Images require a multimodal LLM.
        Args:
            prompt (str): The prompt to use for the question.
            images (list, optional): The images to include in the prompt. Each image must be a path to an image or base64 data. Defaults to None.
            history (list, optional): The history of the conversation. Defaults to None.
        """

        if llm_has_ask(self.llm) is False:
            return None
        
        prompts = [self.system_prompt]
        if history is not None:
            prompts.extend(history)

        prompts.append(make_prompt("user", prompt, images))
        self.chat_history.append(prompts[-1])
        
        response = self.llm.ask(prompts, temperature=self.temperature, format=self.format, stream=self.stream, **self.llm_options)
    
        if self.stream is True:
            return self.handle_stream_response(response, self.chat_history)
        
        self.chat_history.append(make_prompt("assistant", response))
        return response

    
    def handle_stream_response(self, response, chat_history):
        if response.status_code != 200:
            return f"Error: {response.status_code}, {response.text}"
        
        chunks = []

        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    line_text = line_text[6:]
                
                if line_text == "[DONE]":
                    break
                    
                try:
                    chunk = json.loads(line_text)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            chunks.append(delta['content'])
                except json.JSONDecodeError:
                    return f"Error: JSONDecodeError"
        if chat_history is not None:
            complete_resp = "".join(chunks)
            chat_history.append(make_prompt("assistant", complete_resp))
        return chunks



# ------------------------------------------------------------------------------
# example usage:
llm = MyCustomLLM()
agent = MyCustomAgent(llm, agent_type=AgentType.GENERIC_RESPONDER, stream=True)
res = agent.ask("What is 1 + 4?")

for chunk in res:
    print(chunk, end="", flush=True)



# ------------------------------------------------------------------------------
# setting up server to listen on port 1235


# server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# HOST = '127.0.0.1'  # Localhost
# PORT = 1235 
# server_socket.bind((HOST, PORT)) 
# server_socket.listen(1)
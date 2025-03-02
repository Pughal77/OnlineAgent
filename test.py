import json
from llm_axe import Agent, AgentType
import requests

# here it is better to use requests module for sending synchronous http requests as we would want our model to recieve requests in order when it comes to the same chat 

# To use llm-axe with your own LLM, all you need is a class with an ask function
# Example:
class MyCustomLLM():

    # Your ask function will always receive a list of prompts
    # The prompts are in open ai prompt format
    #  example: {"role": "system", "content": "You are a helpful assistant."}
    # If your model supports json format, use the format parameter to specify that to your model.
    def ask(self, prompts:list, format:str="", temperature:float=0.7, stream:bool=False):
        """
        Args:
            prompts (list): A list of prompts to ask.
            format (str, optional): The format of the response. Use "json" for json. Defaults to "".
            temperature (float, optional): The temperature of the LLM. Defaults to 0.8.

        He we are to redirect requests to this LLM to the LLM hosted and listening on a certain port
        """
        data = {
            "model": "deepseek-r1-distill-qwen-7b", # ADD THE NAME OF YOUR MODEL HERE
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

        if stream:
            return self._handle_stream_response(response)
        else:
            if response.status_code != 200:
                return f"Error: {response.status_code}, {response.text}"
            return response.json()['choices'][0]['message']['content']
        
    def _handle_stream_response(self, response):
        if response.status_code != 200:
            return f"Error: {response.status_code}, {response.text}"
            
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
                            yield delta['content']
                except json.JSONDecodeError:
                    return f"Error: JSONDecodeError"



# ------------------------------------------------------------------------------
# example usage:
llm = MyCustomLLM()
agent = Agent(llm, agent_type=AgentType.GENERIC_RESPONDER, stream=True)
print(agent.ask("Hi how are you today?"))
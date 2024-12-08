import requests
from langchain.llms import LLM

class GroqLLM(LLM):
    def __init__(self, api_key, model="llama3-8b-8192", temperature=1, max_tokens=1024, top_p=1, stream=True, stop=None):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.stop = stop

    def _call(self, prompt: str) -> str:
        # Groq API endpoint
        url = "https://api.groq.com/v1/chat/completions"

        # Setting up headers (assuming API key authentication)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Setting up the request body
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
            "stop": self.stop
        }

        # Send the request to the API
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error: {response.status_code}, {response.text}"

    def generate(self, prompt: str) -> str:
        return self._call(prompt)
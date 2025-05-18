import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-07-01-preview", # Specify a supported API version
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

class BasicAzureAgent:
    def __init__(self):
        self.client = client
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        # Define a system prompt to guide the AI's behavior and personality
        self.system_prompt = """
        You are a helpful AI assistant with extensive knowledge about Azure services and DevOps practices.
        Your purpose is to provide clear, accurate information and suggestions when asked questions.
        """

    def ask_question(self, question: str):
        """Sends a question to the Azure OpenAI model and returns the response."""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name, # Specifies the model deployment to use
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7, # Controls randomness: lower is more deterministic
                max_tokens=1000  # Maximum length of the response
            )
            return response.choices[0].message.content
        except Exception as e:
            # Provides error feedback if the API call fails
            return f"Error asking question: {str(e)}"

if __name__ == "__main__":
    agent = BasicAzureAgent()
    print("Azure Agent ready. Type 'quit', 'exit', or 'q' to exit.")
    while True:
        try:
            user_input = input("Your question: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            response_text = agent.ask_question(user_input)
            print("\nResponse:")
            print(response_text)
            print("\n" + "-"*50 + "\n")
        except KeyboardInterrupt: # Allows graceful exit with Ctrl+C
            print("\nExiting agent...")
            break
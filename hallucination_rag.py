import os
import logging
from langchain.tools import Tool
from langchain.llms import CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class HallucinationRAG():

    def __init__(self) -> None:

      """
      Initializes the HallucinationRAG class.

      This constructor sets up the necessary components for the system, including the language model (LLM)
      and user query.

      Parameters:
          None

      Returns:
          None
      """

      # Initialize the language model (LLM) using the CTransformers class
      self.llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_file = 'llama-2-7b-chat.ggmlv3.q2_K.bin', callbacks=[StreamingStdOutCallbackHandler()])
      
      # Set the default user query for the system
      self.user_query = "impact of Ms dhoni’s moon landing"

      # Set Google Search API credentials using environment variables
      os.environ["GOOGLE_CSE_ID"] = "YOUR_GOOGLE_SEARCH_ENGINE_ID"
      os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

      # Set up logging
      logging.basicConfig(filename='hallucination_rag.log', level=logging.INFO)


    def model_response_without_RAG(self) -> str:

        """
        Generates a model response without using the RAG approach.

        This method uses a predefined template and the language model to generate a response.

        Parameters:
            None

        Returns:
            str: The generated response.
        """

        # Define the template for the response
        template = """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Your answers are always brief.
        <</SYS>>
        {text}[/INST]
        """

        # Create a PromptTemplate instance with the template and input variable
        prompt = PromptTemplate(template=template, input_variables=["text"])

        # Create an LLMChain instance with the prompt and language model
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        try:
            # Run the LLMChain with the user's query to generate a response
            result = llm_chain.run(self.user_query)
            logging.info("Generated response without RAG approach.")

            # Return the generated response
            return result

        except Exception as e:
            # Handle the exception by logging an error message
            logging.error(f"Error occurred in model_response_without_RAG: {str(e)}")
            return "An error occurred while generating the response."

    def _set_search_tool(self) -> Tool:

        """
        Creates and returns a Google Search Tool.

        Returns:
            Tool: The Google Search Tool.
        """
        try:
            # Create an instance of the GoogleSearchAPIWrapper to handle Google Search API interactions.
            search = GoogleSearchAPIWrapper()

            # Create a Tool instance with a name, description, and the search.run function as its operation.
            tool = Tool(
                name="Google Search",
                description="Search Google for recent results.",
                func=search.run,
            )

            # Return the Tool instance representing the Google Search tool.
            return tool

        except Exception as e:
            logging.error(f"Error occurred while setting up the search tool: {str(e)}")
            return None


    def _run_tool(self) -> str:

        """
        Executes the search tool to retrieve content from Google.

        Returns:
            str: The content retrieved from Google.
        """

        # Set up the Google Search Tool.
        tool = self._set_search_tool()

        try:

          # Attempt to run the tool with the user's query to retrieve content from Google.
          content = tool.run(self.user_query)

          return content

        except Exception as e:

          # Handle the exception here, you can print an error message or raise a custom exception
          logging.error(f"Error occurred while running the search tool: {str(e)}")
          return ""

    def model_response_RAG_approach(self) -> str:

        """
        Generates a model response using the RAG approach.

        This method retrieves content using the _run_tool method, then processes the content using
        a predefined template and the language model.

        Parameters:
            None

        Returns:
            str: The generated response using the RAG approach.
        """

        # Retrieve content using the _run_tool method.
        content = self._run_tool()

        # Create the template and LLMChain
        template = """
        [INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Analayze the content and answer the user question.
        <</SYS>>
        {content}
        Question:"impact of Ms dhoni’s moon landing"[/INST]
        """

        # Set up a PromptTemplate with the template and input variables.
        prompt = PromptTemplate(template=template, input_variables=["content"])

        # Create an LLMChain with the prompt and the language model.
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)

        # Use the LLMChain to process the content
        try:

          # Use the LLMChain to process the content and generate a final response.
          final_responce = llm_chain.run(content)
          print("With RAG Approach:\n",final_responce)

          return final_responce

        except Exception as e:

          # Handle the exception here, you can print an error message or raise a custom exception
          logging.error(f"Error occurred while generating response with RAG approach: {str(e)}")
          return ""

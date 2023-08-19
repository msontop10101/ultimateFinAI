# finance_app/views.py

from rest_framework.decorators import api_view
from rest_framework.response import Response

# Import the necessary classes and functions from your existing code here


import os



from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.utilities import GoogleSerperAPIWrapper
import re

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
import os

llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

"""Set up tool

"""

# Define which tools the agent can use to answer user queries
search = GoogleSerperAPIWrapper()
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="useful for when you need to ask with search",
    )
]

"""Prompt Template"""

template_with_history = """
Answer the following questions as best you can.

You have access to the following tools: 

{tools}

Use this format:

Question: {input}
Thought: you should think about what to do
Action: the action to take, one of [{tool_names}]  
Action Input: the input to the action
Observation: the result of the action
...(this can repeat N times)
Thought: I now know the final answer
Final Answer: the answer to the original question

Previous conversational history:
{history}

New question: {input}

{agent_scratchpad}
"""

# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        
        # Get the tools
        tools = self.tools
        
        # Create a tools string 
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        
        # Set the tools variable in kwargs
        kwargs["tools"] = tools_str
        
        # Create a list of tool names
        tool_names = [tool.name for tool in tools]
        kwargs["tool_names"] = ", ".join(tool_names)

        # Get the intermediate steps 
        intermediate_steps = kwargs.pop("intermediate_steps")
        
        # Format the intermediate steps
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
            
        # Set the agent_scratchpad variable
        kwargs["agent_scratchpad"] = thoughts
        
        # Format the template string
        return self.template.format(**kwargs)

prompt_with_history = CustomPromptTemplate(
    template=template_with_history,
    tools=tools,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps", "history"]
)

"""Output Parser"""

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Check if the input is not related to finance
        if "Sorry, that's not about finance!" in llm_output:
            return AgentFinish(
                return_values={"output": "Sorry, that's not about finance!"},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # Return an appropriate response instead of raising an error
            return AgentFinish(
                return_values={"output": "Sorry, I could not parse the LLM output."},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser = CustomOutputParser()

output_parser = CustomOutputParser()

"""Set up LLM"""

llm = OpenAI(temperature=0)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt_with_history)

"""Set up the Agent"""

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

memory = ConversationBufferWindowMemory(k=2)

"""Use the Agent"""

# agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# # agent_executor.run("what is the current price of ETH?")

# """Use the Agent"""

# # Create a loop to continuously ask for user input
# while True:
#     # Ask the user for input
#     user_input = input("Ask a question: ")

#     # Check if the user wants to exit the loop
#     if user_input.lower() == "exit":
#         break

#     # Run the agent with user input
#     agent_executor.run(user_input)

# # Print a message when exiting the loop
# print("Exiting the program")



@api_view(['POST'])
def answer_question(request):
    user_input = request.data.get('message')
    
    if user_input is None:
        return Response({"error": "Please provide a 'message' in the request data."}, status=400)
    
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)
    result = agent_executor.run(user_input)
    
    return Response({"answer": result})

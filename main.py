from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()


class Source(BaseModel):
    """Use this class for the sources you used"""

    ans: str = Field(description="A source which was used")


class SearchResult(BaseModel):
    """Use this class as your output format"""

    ans: str = Field(description="The answer for the query")
    sources: list[Source] = Field(
        default_factory=list,
        description="A list of all sources used to generate the answer",
    )


def main():
    llm = ChatOpenAI(model="gpt-4o-mini")
    tools = [TavilySearch()]
    agent = create_agent(llm, tools, response_format=SearchResult)

    query = """
    Give me 3 current job listings for AI Engineering Roles in Germany on LinkedIn, I want to know:
    1. The role
    2. The company
    3. The location
    4. The salary
    """
    print(f"Query: {query}\n")
    result = agent.invoke({"messages": query})
    print(result["structured_response"].ans)


if __name__ == "__main__":
    main()

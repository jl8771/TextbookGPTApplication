from typing import TypedDict, Dict, List, Optional, Any
from openai import OpenAI
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
    
search_tool = DuckDuckGoSearchRun() #Replace with Tavily search?

def summarize_page(page: str) -> str:
    print("Summarized page with tool.")
    client = OpenAI()
    model = "gpt-4o"
    system_prompt = f"""
    You are an expert in textbook summarization for an academic environment. 
    Provide a concise summary highlighting the key points and concepts.
    Your summary should be clear and informative, including the page number.
    
    You may be given a title and page number of the textbook along with the page content.
    The title may contain course codes and author(s).
    
    A course code is 4 capital letters such as SYSC or ELEC followed by 4 digits such as 3002.
    An example of a course code is SYSC3002 or SYSC 3002.
    
    Any mentions of course codes and author(s) in the title should be omitted when referencing the title.
    For example, if the title is "SYSC3002 Software Engineering 5th edition, John Doe",
    you should reference it as "Software Engineering 5th edition".
    """
    content_prompt = f"""
    The page content is as follows:\n{page}
    
    ###
    Generate a summary that captures the essence of the page content in a clear and informative manner, 
    including the page number and title of the textbook.
    Omit the course code and author(s) if present in the title.
    """
    
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": content_prompt}
        ]
    )
    summary = response.output_text.strip()
    
    return summary
    
summarize_page_tool = Tool(
    name="summarize_page",
    func=summarize_page,
    description="Summarizes the content of a textbook page, highlighting key points and concepts."
)
    
def check_blank_page(page: str) -> str:
    print("Checked if page is blank or non-informative with tool.")
    client = OpenAI()
    model = "gpt-4o"
    prompt = f"""
    You are an expert in identifying blank or non-informative pages in academic textbooks.
    You may be given a title and page number of the textbook along with the page content.
    
    A blank page is defined as a page that only contains whitespace or unicode characters
    that do not convey any meaningful information. A page may contain headers, footers,
    and page numbers but still be considered blank if there is no substantive content.
    
    A non-informative page is defined as a page that contains very little or no useful information.
    This could be pages with title pages, copyright information, dedication pages, or title of contents.
    These pages contain content but are not useful for learning or understanding the subject matter.
    
    Given the following page, determine if it is blank or non-informative using only a single word answer,
    one of "Blank", "Non-informative", or "Useful".
    
    Below is the text to analyze:
    
    ###
    {page}
    """
    
    response = client.responses.create(
        model=model,
        input=[
            {"role": "user", "content": prompt}
        ]
    )
    
    page_category = response.output_text.strip()
    print(f"Categorized page category {page_category}")
    if "Blank" in page_category:
        is_blank = True
        is_useful = False
    elif "Non-informative" in page_category:
        is_blank = False
        is_useful = False
    elif "Useful" in page_category:
        is_blank = False
        is_useful = True
    else:
        is_blank = "Unknown"
        is_useful = "Unknown"
    
    return f"Is blank: {is_blank}, Is useful: {is_useful}"

check_blank_page_tool = Tool(
    name="check_blank_page",
    func=check_blank_page,
    description="Checks if a textbook page is blank, non-informative, or useful."
)
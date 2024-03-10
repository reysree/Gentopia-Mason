from typing import Type
from pydantic import BaseModel, Field
from gentopia.tools.basetool import BaseTool
from scholarly import scholarly
import requests
import fitz  # PyMuPDF for PDF handling
from transformers import pipeline  # For text summarization
import os
import asyncio

class PdfSummarizeArgs(BaseModel):
    title: str = Field(..., description="Title of the paper to search on Google Scholar and summarize.")

class PdfSummarize(BaseTool):
    """Tool to search for a paper on Google Scholar, download the PDF, and summarize it."""
    
    name = "pdf_summarize"
    description = "Searches for a paper on Google Scholar, downloads it, and summarizes the content."
    args_schema: Type[BaseModel] = PdfSummarizeArgs

    def _run(self, title: str) -> str:
        paper_search_tool = scholarly.search_single_pub(title)
        pdf_url = paper_search_tool.get('eprint_url', '')
        
        if not pdf_url:
            return "PDF URL not found."
        
        #save_directory = "/home/sreeram/Gentopia-Mason/Gentopia/gentopia/tools"
        current_directory = os.getcwd()
        #os.makedirs(save_directory, exist_ok=True)
        safe_title = "".join(x for x in title.replace(" ", "_") if x.isalnum() or x == "_")
        pdf_filename = os.path.join(current_directory, safe_title + ".pdf")
        try:
            #Downloading paper
            response = requests.get(pdf_url)
            response.raise_for_status()
            with open(pdf_filename, 'wb') as f:
                f.write(response.content)
            extracted_text = self.extract_text_from_pdf(pdf_filename)
            summary = self.summarize_text(extracted_text)
            return summary
        except requests.exceptions.HTTPError as err:
            return f"HTTP Error: {err}"
        except Exception as e:
            return f"An error occurred: {e}"

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        doc = fitz.open(pdf_path)
        text = ''
        abstract_found = False
        introduction_found = False
        conclusion_found = False

        for page in doc:
            page_text = page.get_text()
            # Simple checks to find sections; could be refined based on common patterns
            if 'abstract' in page_text.lower():
                abstract_found = True
                text += "\n" + page_text
            elif 'introduction' in page_text.lower():
                introduction_found = True
                text += "\n" + page_text
            elif 'conclusion' in page_text.lower():
                conclusion_found = True
                text += "\n" + page_text
                break  # Assuming conclusion is the last section of interest

        doc.close()
        return text
    
    async def _arun(self, title: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run, title)


    def summarize_text(self, text: str) -> str:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        # Split the text into chunks of appropriate size
        words = text.split()
        chunk_size = 100  # Adjust based on your needs and model's limitations
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

        summarized_text = []
        for chunk in chunks:
            # Adjust max_length and min_length dynamically based on chunk size
            input_length = len(chunk.split())
            max_length = min(130, input_length // 2)  # Adjust as needed
            min_length = max(30, max_length // 4)  # Adjust as needed

            # Summarize the chunk
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summarized_text.append(summary[0]['summary_text'])

        # Combine the summaries of all chunks
        final_summary = ' '.join(summarized_text)
        return final_summary

"""
title: Confluence RAG Pipe
author: James W. (0xThresh)
author_url: https://github.com/0xThresh
funding_url: https://github.com/open-webui
version: 0.1
license: MIT
requirements: atlassian-python-api, pytesseract, Pillow, langchain==0.3, langchain_community==0.3, langchain_ollama==0.2.0, pydantic
"""

import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from typing import Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import get_last_user_message


class Pipe:
    class Valves(BaseModel):
        CONFLUENCE_SITE: str = Field(
            description="The URL of your Confluence site, such as https://open-webui.atlassian.net/",
        )
        pass

    class UserValves(BaseModel):
        CONFLUENCE_SPACE: str = Field(
            description="The Confluence space to pull pages from",
        )
        CONFLUENCE_USERNAME: str = Field(
            description="Your Confluence username for the site you pull documents from",
        )
        CONFLUENCE_API_KEY: str = Field(
            description="Your Confluence API key, generated here for Confluence Cloud: https://id.atlassian.com/manage-profile/security/api-tokens",
        )
        OLLAMA_MODEL: str = Field(
            default="llama3:instruct", description="The Ollama model to use"
        )
        pass

    def __init__(self):
        self.type = "pipe"
        self.id = "Confluence RAG"
        self.name = "confluence_rag"
        self.valves = self.Valves(
            **{
                "CONFLUENCE_SITE": os.getenv("CONFLUENCE_SITE", ""),
            }
        )
        pass

    def get_confluence_docs(self, username, api_key, space_key):
        loader = ConfluenceLoader(
            url=self.valves.CONFLUENCE_SITE,
            username=username,
            api_key=api_key,
            space_key=space_key,
            include_attachments=False,  # This is failing, try to fix later
            # Start by limiting to 5 to avoid overwhelming the context window
            max_pages=5,
        )
        documents = loader.load()
        return documents

    def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __event_call__=None,
        __task__=None,
        __task_body__: Optional[dict] = None,
        __valves__=None,
    ) -> Union[str, Generator, Iterator]:

        user_valves = __user__.get("valves")
        print("USER VALVES:")
        print(user_valves)

        if not user_valves:
            raise Exception("User Valves not configured.")

        # Get the documents from Confluence
        documents = self.get_confluence_docs(
            user_valves.CONFLUENCE_USERNAME,
            user_valves.CONFLUENCE_API_KEY,
            user_valves.CONFLUENCE_SPACE,
        )

        # Get the user's question from the UI
        query = get_last_user_message(body["messages"])

        # split docs into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        splits = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Load documents into Chroma
        vectorstore = Chroma.from_documents(splits, embedding_function)
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )
        retrieved_docs = retriever.invoke(query)
        print("-------RETRIEVED DOCS-------")
        print(retrieved_docs)
        prompt = hub.pull("rlm/rag-prompt", api_key="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")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Initialize model
        llm = OllamaLLM(model=user_valves.OLLAMA_MODEL)
        print("------DEBUG: MODEL--------")
        print(llm)
        #print(type(query))

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print(rag_chain.invoke(query))

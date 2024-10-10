"""
title: Confluence Pipe
author: James W. (0xThresh)
author_url: https://github.com/0xThresh
funding_url: https://github.com/open-webui
version: 0.1
license: MIT
requirements: atlassian-python-api, pytesseract, Pillow
"""

import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_text_splitters import CharacterTextSplitter
from typing import Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
from open_webui.utils.misc import get_last_user_message


class Pipe:
    class Valves(BaseModel):
        CONFLUENCE_SITE: str = Field(
            description="The URL of your Confluence site, such as https://open-webui.atlassian.net/",
        )

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
            include_attachments=True,
            # Start by limiting to 5 to avoid overwhelming the context window
            limit=5,
        )
        documents = loader.load()
        return documents

    def pipes(self):
        return [
            {
                "id": self.id,
                "name": self.name,
            }
        ]

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
            user_valves.CONFLUENCE_SPACE
        )

        # split docs into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Load documents into Chroma
        db = Chroma.from_documents(docs, embedding_function)
        print("---------DB----------")
        print(db)
        print(type(db))

        # Get the query from the user and query Chroma with it
        query = get_last_user_message(body["messages"])
        print("------QUERY------")
        print(query)
        docs = db.similarity_search(query)

        # Return response
        print(docs[0].page_content)
        response = query_engine.query(user_message)
        print(docs)

        return docs[0].page_content

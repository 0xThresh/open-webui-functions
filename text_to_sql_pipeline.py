"""
title: Llama Index DB Pipe
author: James W. (0xThresh)
author_url: https://github.com/0xThresh
funding_url: https://github.com/open-webui
version: 1.3
requirements: llama_index, sqlalchemy, mysql-connector, psycopg2-binary, llama_index.llms.ollama
"""

from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator, Optional
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase, PromptTemplate
import llama_index.core
from sqlalchemy import create_engine
from open_webui.utils.misc import get_last_user_message
import logging


class Pipe:
    class Valves(BaseModel):
        DB_ENGINE: str = Field(
            os.getenv("DB_TYPE", "postgres"),
            description="Database type (supports 'postgres' and 'mysql', defaults to postgres)",
        )
        DB_HOST: str = Field(
            os.getenv("DB_HOST", "host.docker.internal"),
            description="Database hostname",
        )
        DB_PORT: str = Field(
            os.getenv("DB_PORT", "5432"), description="Database port (default: 5432)"
        )
        DB_USER: str = Field(
            os.getenv("DB_USER", "postgres"),
            description="Database user to connect with. Make sure this user has permissions to the database and tables you define",
        )
        DB_PASSWORD: str = Field(
            os.getenv("DB_PASSWORD", "password"), description="Database user's password"
        )
        DB_DATABASE: str = Field(
            os.getenv("DB_DATABASE", "postgres"),
            description="Database with the data you want to ask questions about",
        )
        DB_TABLE: str = Field(
            os.getenv("DB_TABLE", "table"),
            description="Table in the database with the data you want to ask questions about",
        )
        OLLAMA_HOST: str = Field(
            os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
            description="Hostname of the Ollama host with the model",
        )
        TEXT_TO_SQL_MODEL: str = Field(
            os.getenv("TEXT_TO_SQL_MODEL", "llama3.2:3b"),
            description="LLM model to use for text-to-SQL operation. You may need to experiment with different models for best results",
        )
        CONTEXT_WINDOW: int = Field(
            os.getenv("CONTEXT_WINDOW", 30000),
            description="The number of tokens to use in the context window",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.log = logging.getLogger(__name__)
        # Update or comment out log level as needed
        self.log.setLevel(logging.DEBUG)
        self.type = "pipe"
        self.name = "Database RAG Pipeline"
        self.engine = None
        self.nlsql_response = ""
        pass

    def init_db_connection(self):
        # Update your DB connection string based on selected DB engine - current connection string is for Postgres
        if self.valves.DB_ENGINE == "mysql":
            self.engine = create_engine(
                f"mysql+mysqlconnector://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}"
            )
        elif self.valves.DB_ENGINE == "postgres":
            self.engine = create_engine(
                f"postgresql+psycopg2://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}"
            )
        else:
            raise ValueError(f"Unsupported database engine: {self.valves.DB_ENGINE}")

        return self.engine

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __event_call__=None,
        __task__=None,
        __task_body__: Optional[dict] = None,
        __valves__=None,
    ) -> Union[str, Generator, Iterator]:
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Connecting to the database...", "done": False},
            }
        )

        print(f"pipe:{__name__}")

        print(__event_emitter__)
        print(__event_call__)

        # Create database reader for Postgres
        self.init_db_connection()
        sql_database = SQLDatabase(
            self.engine, include_tables=self.valves.DB_TABLE.split(",")
        )
        await __event_emitter__(
            {
                "type": "status",  # We set the type here
                "data": {"description": "Querying the database...", "done": False},
            }
        )

        # Set up LLM connection
        llm = Ollama(
            model=self.valves.TEXT_TO_SQL_MODEL,
            base_url=self.valves.OLLAMA_HOST,
            request_timeout=300.0,
            context_window=self.valves.CONTEXT_WINDOW,
        )

        # Set up the custom prompt used when generating SQL queries from text
        text_to_sql_prompt = """
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
        You can order the results by a relevant column to return the most interesting examples in the database.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per the database engine. You can order the results to return the most informative data in the database.
        Never query for all the columns from a specific table, only ask for a few relevant columns given the question.
        You should use DISTINCT statements and avoid returning duplicates wherever possible.
        Do not return example data if no data is found. You have access to a SQL database, and the only results you should return are values from that database.
        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        Only use tables listed below.
        {schema}

        Question: {query_str}
        SQLQuery: 
        """

        text_to_sql_template = PromptTemplate(text_to_sql_prompt)

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=self.valves.DB_TABLE.split(","),
            llm=llm,
            text_to_sql_prompt=text_to_sql_template,
            embed_model="local",
            streaming=True,
        )

        user_message = get_last_user_message(body["messages"])

        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": "Returning response...", "done": True},
            }
        )

        response = query_engine.query(user_message)

        return response.response_gen

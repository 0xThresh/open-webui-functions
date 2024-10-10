"""
title: Llama Index DB Pipe
author: James W. (0xThresh)
author_url: https://github.com/0xThresh
funding_url: https://github.com/open-webui
version: 1.0
requirements: llama_index, sqlalchemy, psycopg2-binary, llama_index.llms.ollama
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

# If you need to see where the SQL query is failing, uncomment the line below
# llama_index.core.set_global_handler("simple")


class Pipe:
    class Valves(BaseModel):
        DB_HOST: str = Field(
            os.getenv("DB_HOST", "http://localhost"), description="Database hostname"
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
            os.getenv("DB_TABLE", "table_name"),
            description="Table in the database with the data you want to ask questions about",
        )
        OLLAMA_HOST: str = Field(
            os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
            description="Hostname of the Ollama host with the model",
        )
        TEXT_TO_SQL_MODEL: str = Field(
            os.getenv("TEXT_TO_SQL_MODEL", "phi3:latest"),
            description="LLM model to use for text-to-SQL operation. Note that some models fail at SQL generation, such as llama3.2",
        )
        pass

    # TODO: implement in the next version to allow individuals to connect to any DB
    # class UserValves(BaseModel):
    #     DB_HOST: str = Field(
    #         os.getenv("DB_HOST", "http://localhost"), description="Database hostname"
    #     )
    #     DB_PORT: str = Field(
    #         os.getenv("DB_PORT", "5432"), description="Database port (default: 5432)"
    #     )
    #     DB_USER: str = Field(
    #         os.getenv("DB_USER", "postgres"),
    #         description="Database user to connect with. Make sure this user has permissions to the database and tables you define",
    #     )
    #     DB_PASSWORD: str = Field(
    #         os.getenv("DB_PASSWORD", "password"), description="Database user's password"
    #     )
    #     DB_DATABASE: str = Field(
    #         os.getenv("DB_DATABASE", "postgres"),
    #         description="Database with the data you want to ask questions about",
    #     )
    #     DB_TABLE: str = Field(
    #         os.getenv("DB_TABLE", "table_name"),
    #         description="Table in the database with the data you want to ask questions about",
    #     )
    #     OLLAMA_HOST: str = Field(
    #         os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"),
    #         description="Hostname of the Ollama host with the model",
    #     )
    #     TEXT_TO_SQL_MODEL: str = Field(
    #         os.getenv("TEXT_TO_SQL_MODEL", "phi3:latest"),
    #         description="LLM model to use for text-to-SQL operation",
    #     )
    #     pass

    def __init__(self):
        self.valves = self.Valves()
        self.name = "Database RAG Pipeline"
        self.engine = None
        self.nlsql_response = ""
        pass

    def init_db_connection(self):
        # Update your DB connection string based on selected DB engine - current connection string is for Postgres
        self.engine = create_engine(
            f"postgresql+psycopg2://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}"
        )
        return self.engine

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

        print(f"pipe:{__name__}")

        print(__event_emitter__)
        print(__event_call__)

        if __task__:
            print(__task__)
            if __task__ == "title_generation":
                # TODO: Fix this to use the generated title
                return "Text to SQL 💽"

        # Create database reader for Postgres
        self.init_db_connection()
        sql_database = SQLDatabase(self.engine, include_tables=[self.valves.DB_TABLE])

        # Set up LLM connection; uses phi3 model with 128k context limit since some queries have returned 20k+ tokens
        llm = Ollama(
            model=self.valves.TEXT_TO_SQL_MODEL,
            base_url=self.valves.OLLAMA_HOST,
            request_timeout=180.0,
            context_window=30000,
        )

        # Set up the custom prompt used when generating SQL queries from text
        text_to_sql_prompt = """
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
        You can order the results by a relevant column to return the most interesting examples in the database.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per Postgres. You can order the results to return the most informative data in the database.
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
            tables=[self.valves.DB_TABLE],
            llm=llm,
            text_to_sql_prompt=text_to_sql_template,
            embed_model="local",
            streaming=True,
        )

        user_message = get_last_user_message(body["messages"])

        response = query_engine.query(user_message)

        return response.response_gen

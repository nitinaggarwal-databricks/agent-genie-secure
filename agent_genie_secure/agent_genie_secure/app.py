from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import logging
import asyncio
import uvicorn
from dotenv import load_dotenv
import os
import re
import io
import uuid
from typing import Optional, Any, List
from pydantic import BaseModel

# PDF extraction
from pypdf import PdfReader

# Load the .env file
load_dotenv()

##cfg = Config()

# Import the updated helper functions
from helper import  fetch_answer
from table_extraction import get_tables, get_table_columns
from tracking import create_user_interaction_table, log_user_interaction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# Initialize Databricks client
client = WorkspaceClient()

# Load GENIE_ROOM_ID from environment variable early
GENIE_ROOM_ID_FROM_ENV = os.getenv("SPACE_ID")

# Load SERVING_ENDPOINT_NAME from environment variable
SERVING_ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT_NAME")

# Validate SERVING_ENDPOINT_NAME is set
if not SERVING_ENDPOINT_NAME:
    raise ValueError("SERVING_ENDPOINT_NAME environment variable is not set. Please configure it in your environment.")

# Global variables
CURRENT_CONVERSATION_ID = None
DYNAMIC_GENIE_ROOM_ID = GENIE_ROOM_ID_FROM_ENV  # Initialize from environment variable

# --- Simple in-memory session store (swap for Redis in prod) ---
SESSION_STORE: dict[str, str] = {}  # session_id -> pdf_content
SCHEMA_INFO = ""  # Global variable to store schema information


from manual_ai_content import MANUAL_AI_CONTENT
# Get configuration from Databricks SDK
from databricks.sdk.core import Config
cfg = Config()

# Load environment variables
fallback_workspace_url = "https://" + cfg.hostname
fallback_access_token = cfg.oauth_token().access_token

# FIX: proper logging formatting so host actually appears in logs
logger.info("hostname %s", fallback_workspace_url)

# =========================
# Response Parsing Helpers
# =========================
def _content_to_text(message_content: Any) -> str:
    """
    Normalize Databricks/LLM message.content which may be:
      - str
      - list of parts (each a str or dict with keys like {'type': 'text'|'output_text', 'text': '...'})
      - object with a .text property
    """
    if message_content is None:
        return ""
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        parts: List[str] = []
        for p in message_content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if "text" in p and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif "value" in p and p.get("value") is not None:
                    parts.append(str(p["value"]))
            else:
                txt = getattr(p, "text", None)
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(t for t in parts if t)

    return str(message_content)


def _extract_message_content(resp: Any) -> str:
    """
    Pull content out of various response shapes from Databricks Serving:
      - Common: resp.choices[0].message.content
      - Future/alt: resp.output_text, or resp.output[0].content
    """
    raw = None
    try:
        raw = resp.choices[0].message.content
    except Exception:
        pass

    if raw is None:
        raw = getattr(resp, "output_text", None)

    if raw is None:
        out = getattr(resp, "output", None)
        if isinstance(out, list) and out:
            raw = getattr(out[0], "content", None)

    return _content_to_text(raw).strip()


def determine_required_columns(query, table_schema):
    """
    Analyze the user query and determine which columns from the schema are required
    
    Args:
        query (str): The user's query
        table_schema (dict): The schema of available tables with their columns
        
    Returns:
        list: A list of required column names
    """
    global SCHEMA_INFO
    
    # Build schema_info from the passed table_schema parameter
    schema_info = ""
    
    if table_schema and isinstance(table_schema, dict) and len(table_schema) > 0:
        # Use the actual table schema passed to the function
        logger.info(f"Using passed table_schema with {len(table_schema)} tables")
        for table_name, columns in table_schema.items():
            schema_info += f"Table: {table_name}\n"
            if columns and len(columns) > 0:
                schema_info += f"Columns: {', '.join(columns)}\n"
            else:
                schema_info += "Columns: [No column information available]\n"
            schema_info += "\n"
        
        # Also extract all unique column names for easier processing
        all_columns = set()
        for columns in table_schema.values():
            if columns:
                all_columns.update(columns)
        
        if all_columns:
            schema_info += f"\nAll Available Columns: {', '.join(sorted(all_columns))}\n"
            
    elif SCHEMA_INFO:
        # Use the global SCHEMA_INFO if table_schema is empty but SCHEMA_INFO is available
        logger.info("Using global SCHEMA_INFO as fallback")
        schema_info = f"Available Columns: {SCHEMA_INFO}"
    else:
        # Use hardcoded fallback as last resort
        logger.warning("Using hardcoded fallback schema")
        schema_info = """Available Columns: order_id, order_datetime, abnormal_flag, first_name, diagnosis_type, message_type, value, address, guarantor_phone, diagnosis_code, discharge_datetime, reference_range, ordering_provider, test_name, recorded_datetime, diagnosis_description, source_file, guarantor_address, sending_fac, phone, guarantor_name, admit_datetime, unit, receiving_app, attending_doctor, observation_id, assigned_location, last_name, patient_class, message_datetime, hl7_version, sending_app, dob, gender, patient_id, event_type"""
    
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content=f"""
You are a Business Analytics and AI expert in healthcare data analysis. Your task is to review the available database schema and the user's query, and identify only the column names required to answer the query.

Database Schema Information:
{schema_info}

Instructions:
- Analyze the user's query and identify which columns from the available schema are needed to answer it
- Return a JSON array containing ONLY the column names needed to answer the query, formatted as:
  ["column1", "column2", "column3", ...]
- Only include columns that are directly necessary for answering the query
- For patient-related queries, include relevant patient identifiers (patient_id) and demographic columns (first_name, last_name, dob, gender)
- For encounter/admission queries, include encounter identifiers and related columns (admit_datetime, discharge_datetime, attending_doctor, assigned_location)
- For diagnosis queries, include diagnosis-related columns (diagnosis_code, diagnosis_description, diagnosis_type)
- For test/lab queries, include test-related columns (test_name, value, reference_range, abnormal_flag)
- If the query involves aggregations, grouping, or calculations, include the columns required for those operations
- If the query asks for dataset structure or schema information, include representative columns from different categories
- Do NOT specify any table names in your response
- Do NOT add any explanation, description, or extra text—output ONLY the JSON array of column names
- If no specific columns are needed (e.g., for general information queries), return an empty array []

When you receive a query, analyze it against the provided schema and return only the necessary column names in the exact JSON array format.
"""
            ),
            ChatMessage(role=ChatMessageRole.USER, content=query),
        ],
    )
    
    response = _extract_message_content(resp)
    logger.info(f"Required columns for query '{query}': {response}")
    
    # Try to parse the response as JSON
    try:
        import json
        json_str = response.strip()
        start = json_str.find('[')
        end = json_str.rfind(']') + 1
        if start >= 0 and end > start:
            json_str = json_str[start:end]
        required_columns = json.loads(json_str)
        return required_columns
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse required columns response as JSON: {response}")
        logger.error(f"JSON error: {str(e)}")
        return []

def generate_dynamic_questions(schema_info=None):
    """
    Generate dynamic questions based on the provided schema information
    
    Args:
        schema_info (str): Schema information containing column names
        
    Returns:
        list: A list of generated questions based on the schema
    """
    global SCHEMA_INFO
    
    # Use provided schema_info or fall back to global SCHEMA_INFO
    if not schema_info and SCHEMA_INFO:
        schema_info = SCHEMA_INFO
    elif not schema_info:
        schema_info = "No schema information available"
    
    print("Schema information passed to generate_dynamic_questions:")
    print(schema_info)
    
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content=f"""
You are a Data Analyst. Your job is to create simple, plain English questions that can be answered using SQL queries. I will provide you with a sample dataset (column names only), and you will generate 5 simple SQL-friendly questions based on the available columns.

Database Schema Information:
{schema_info}

Instructions:
- Make the questions easy to understand and suitable for beginner to intermediate SQL users.
- Generate 5 simple questions in plain English based on the above columns.
- Return the questions as a list for example: ["Question 1", "Question 2", "Question 3", "Question 4", "Question 5"]
- Focus on common business analytics queries like counts, aggregations, filtering, and basic insights.
- Make sure the questions are relevant to the available columns.

Give me the list of the questions in the list format only. Dont give any other part than generated questions. 
"""
            ),
            ChatMessage(role=ChatMessageRole.USER, content="Generate 5 simple questions based on the provided schema information."),
        ],
    )
    
    response = _extract_message_content(resp)
    print(f"Generated questions response: {response}")
    
    # Try to parse the response as a Python list
    try:
        import ast
        if response.strip().startswith('[') and response.strip().endswith(']'):
            questions = ast.literal_eval(response.strip())
            if isinstance(questions, list):
                return questions
        
        # If not a proper list format, try to extract questions manually
        lines = response.strip().split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                import re
                clean_line = re.sub(r'^\d+\.?\s*', '', line)
                clean_line = re.sub(r'^[-*]\s*', '', clean_line)
                clean_line = clean_line.strip('"\'')
                if clean_line:
                    questions.append(clean_line)
        
        if questions:
            return questions[:5]
            
    except Exception as e:
        logger.error(f"Failed to parse generated questions: {str(e)}")
    
    logger.info("Using predefined sample questions as fallback")
    return [
        "Show me the first 10 rows of the dataset",
        "How many total records are in the dataset?",
        "What are the unique values in the main categories?",
        "Show me summary statistics for the numerical columns",
        "What is the data distribution by key fields?"
    ]

def rephrase_query(query):
    """
    Rephrase the user's query to better understand the intent
    """
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="""
You are a query understanding assistant. Your task to analyze the user's query and rephrase it to clarify it with the same meaning with following rules. 

1. Identify the core information need what columns are required
2. Add any implied context that would help with answering
3. Structure the question in a clear, direct way
4. Return ONLY the rephrased question without any explanations or additional text
5. If forecast keyword present in the query, then add the strictly 'forecast' keyword to the rephrased query
6. If classify keyword present in the query, then add the 'classify' keyword to the rephrased query

Example:
User: "show me patient numbers"
Output: "What is the total count of unique patients in the database?"
"""
            ),
            ChatMessage(role=ChatMessageRole.USER, content=query),
        ],
    )
    rephrased = _extract_message_content(resp)
    logger.info(f"Original query: '{query}' → Rephrased: '{rephrased}'")
    print(f"Rephrased query: {rephrased}")
    return rephrased

def rephrase_query_forecast(query):
    """
    Rephrase the user's query to better understand the intent
    """
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="""
You are a query rephrasing assistant for a forecasting application. Your job is to:
- Read the user's forecast question and determine the correct aggregation level needed (yearly, monthly, weekly, daily).
- If the user asks for a yearly forecast, clarify that you will aggregate monthly data and display the yearly result using DD-MM-YYYY format (e.g., 01-01-2024 for 2024).
- If the user asks for a monthly forecast, clarify that you will aggregate weekly data and display the monthly result using DD-MM-YYYY format (e.g., 01-05-2024 for May 2024).
- If the user asks for a weekly forecast, clarify that you will aggregate daily data and display the weekly result as the week's Monday in DD-MM-YYYY format (e.g., 06-05-2024 for the week starting 6th May 2024).
- If the user asks for a daily forecast, clarify that you will aggregate hourly data and display the daily result using DD-MM-YYYY format (e.g., 08-05-2024 for 8th May 2024).
- Always rephrase the query to be concise, clear, and specific, preserving the original intent and subject.
- Always specify the date format for the results.

Example Inputs and Outputs:

Input: "Show me the yearly forecast for revenue."
Output: "Provide a yearly revenue forecast by aggregating monthly data, and show each year's result as DD-MM-YYYY (e.g., 01-01-2024)."

Input: "I want the monthly forecast for orders."
Output: "Provide a monthly orders forecast by aggregating weekly data, and show each month's result as DD-MM-YYYY (e.g., 01-05-2024 for May 2024)."

Input: "Give the weekly forecast for sales."
Output: "Provide a weekly sales forecast by aggregating daily data, and show each week's result as DD-MM-YYYY (using the Monday of each week, e.g., 06-05-2024)."

Input: "What is the daily forecast for the next 7 days?"
Output: "Provide a daily forecast for the next 7 days by aggregating hourly data, and show each day's result as DD-MM-YYYY."

Always rephrase the user's forecast query to make the aggregation logic and output format clear and actionable.
"""
            ),
            ChatMessage(role=ChatMessageRole.USER, content=query),
        ],
    )
    rephrased = _extract_message_content(resp)
    logger.info(f"Original query: '{query}' → Rephrased: '{rephrased}'")
    print(f"Rephrased query: {rephrased}")
    return rephrased

def classify_query(query):
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="""
You are a highly specialized classification assistant designed to support Databricks SQL workloads. Your sole task is to classify any user question into **exactly one** of the following three categories based on the nature of the query and the intent behind it.

Your classification should be based on the following detailed definitions:

1. **Normal SQL** — Classify here if the user's question can be answered using standard SQL operations (e.g., SELECT, WHERE, JOIN, GROUP BY, ORDER BY, HAVING, COUNT, SUM, AVG, MIN, MAX) on structured tables or views within a Databricks environment. These questions are factual and directly refer to known data stored in a relational schema. Examples:
   - "What is the average length of stay per hospital unit?"
   - "List the top 5 procedures performed last month."
   - "Show all patients with a diagnosis of diabetes."
   - Explain the dataset
   - Give me the row count
   - Show me the schema

2. **Predictive SQL** — Classify here if the question involves machine learning, statistical inference, anomaly detection, translation/translate, pattern mining, classification, forecasting, or any form of predictive analytics. These typically require training or applying a model and go beyond basic SQL capabilities. Examples:
   - "Which patients are at risk of readmission?"
   - "Detect anomalies in test results by provider."
   - "Forecast next likely diagnosis based on history."
   - "summarize the patient history"
   - "Can you translate diagnosis and treatment plans into Spanish for Spanish-speaking members?"
   - "Can we classify members into high, medium, and low risk based on recent encounters?"
   - "How would you translate the complex diagnosis_code into human-readable diagnosis_descriptions?"
   - "What translation services are available to convert medical diagnosis and treatment plans into Spanish for Spanish-speaking patients?"
   
3. **General Information** — Classify here if the question cannot be answered using Databricks SQL on the available structured datasets, or if it requires external/general knowledge such as product descriptions, definitions, APIs, or documentation. These are not queries against data tables. Examples:
   - "What is Databricks Unity Catalog?"
   - "Explain what HL7 messages mean."
   - "Does Databricks support Python for machine learning?"

**Your response must be one of the following strings only:**
- Normal SQL
- Predictive SQL
- General Information

Do **not** provide any explanation, examples, justification, or additional text. Only return the correct category string.
"""
            ),
            ChatMessage(role=ChatMessageRole.USER, content=query),
        ],
    )
    return _extract_message_content(resp)

def classify_predictive_query(query):
    """
    Classify predictive queries into specific AI functions
    Returns a list of AI function names when multiple functions are detected
    """
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content="""
You are a specialized classification assistant for Databricks AI functions. Your task is to classify user queries into one or more of the following AI function categories based on the specific tasks they want to accomplish:

**AI Function Categories:**

1. **ai_analyze_sentiment** — For analyzing sentiment/emotion in text data
   Examples: "What is the sentiment of patient feedback?", "Analyze the sentiment of reviews"

2. **ai_classify** — For general classification tasks with custom labels
   Examples: "Classify patients into risk categories", "Categorize incidents by severity"

3. **ai_extract** — For extracting specific entities or information from text
   Examples: "Extract medication names from notes", "Pull out dates from documents"

4. **ai_fix_grammar** — For correcting grammatical errors in text
   Examples: "Fix grammar in patient notes", "Correct spelling errors in reports"

5. **ai_gen** — For general text generation and answering prompts
   Examples: "Generate a summary", "Create a report", "Answer questions about data"

6. **ai_mask** — For masking/anonymizing specific entities in text
   Examples: "Mask patient names", "Hide sensitive information", "Anonymize PII"

7. **ai_similarity** — For comparing similarity between text strings
   Examples: "Find similar diagnoses", "Compare patient descriptions", "Match similar records"

8. **ai_summarize** — For generating summaries of text or data
   Examples: "Summarize patient history", "Create executive summary", "Summarize findings"

9. **ai_translate** — For translating text between languages
    Examples: "Translate to Spanish", "Convert to French", "Translate diagnosis"

10. **ai_forecast** — For forecasting and time series predictions
    Examples: "Forecast patient visits", "Predict future trends", "Project revenue"

**Instructions:**
- If the query contains multiple tasks, return a JSON array with all relevant AI function names
- If the query contains only one task, return a JSON array with one AI function name
- Return ONLY the JSON array format: ["ai_function_name1", "ai_function_name2", ...]
- Do not provide any explanation, examples, justification, or additional text
- Only return valid AI function names from the list above

**Examples:**
- "Forecast patient visits and summarize the results" → ["ai_forecast", "ai_summarize"]
- "Classify patients and extract medication names" → ["ai_classify", "ai_extract"]
- "Translate diagnosis to Spanish" → ["ai_translate"]
- "find anomalies in the data and give the reason for the anomaly" → ["ai_classify", "ai_gen"]
"""
            ),
            ChatMessage(role=ChatMessageRole.USER, content=query),
        ],
    )
    
    response = _extract_message_content(resp)
    print("advanced classification", response)
    
    # Parse the response to extract AI function names
    try:
        import json
        if response.startswith('[') and response.endswith(']'):
            ai_functions = json.loads(response)
            if isinstance(ai_functions, list):
                return ai_functions
        
        if response in ['ai_analyze_sentiment', 'ai_classify', 'ai_extract', 'ai_fix_grammar', 
                       'ai_gen', 'ai_mask', 'ai_similarity', 'ai_summarize', 'ai_translate', 'ai_forecast']:
            return [response]
        
        import re
        function_names = re.findall(r'ai_\w+', response)
        if function_names:
            return function_names
            
    except Exception as e:
        logger.error(f"Error parsing AI function classification: {str(e)}")
    
    return [response]

async def explain_dataset_directly(table_schema=None, question=None):
    """
    Directly explain the dataset using schema information without classification.
    This function is specifically called when user asks "explain the dataset".
    
    Args:
        table_schema (dict, optional): Table schema information with table names and columns
        question (str, optional): The original question asked by the user
        
    Returns:
        str: Comprehensive explanation of the dataset
    """
    global SCHEMA_INFO, CURRENT_CONVERSATION_ID
    
    try:
        # Prepare schema information
        schema_info = ""
        
        if table_schema and isinstance(table_schema, dict) and len(table_schema) > 0:
            logger.info(f"Using passed table_schema with {len(table_schema)} tables for dataset explanation")
            for table_name, columns in table_schema.items():
                schema_info += f"Table: {table_name}\n"
                if columns and len(columns) > 0:
                    schema_info += f"Columns ({len(columns)}): {', '.join(columns)}\n"
                else:
                    schema_info += "Columns: [No column information available]\n"
                schema_info += "\n"
            
            all_columns = set()
            for columns in table_schema.values():
                if columns:
                    all_columns.update(columns)
            
            if all_columns:
                schema_info += f"Total Unique Columns Across All Tables: {len(all_columns)}\n"
                schema_info += f"Column Names: {', '.join(sorted(all_columns))}\n"
                
        elif SCHEMA_INFO:
            logger.info("Using global SCHEMA_INFO for dataset explanation")
            columns_list = SCHEMA_INFO.split('\t') if SCHEMA_INFO else []
            schema_info = f"Dataset Contains {len(columns_list)} Columns:\n"
            schema_info += f"Column Names: {', '.join(columns_list)}\n"
        else:
            schema_info = "No schema information is currently available for this dataset."
        
        prompt = f"""

=== DATASET SCHEMA INFORMATION ===
{schema_info}

=== USER'S ORIGINAL QUESTION ===
{question}
"""
        
        resp = client.serving_endpoints.query(
            SERVING_ENDPOINT_NAME,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="""YOu are expert in answering the following question based on the schema. Give the answer in less than 100 words. Give me in beautiful summary. 
                    """
                ),
                ChatMessage(role=ChatMessageRole.USER, content=prompt),
            ],
        )
        
        response = _extract_message_content(resp)
        logger.info(f"✅ Generated dataset explanation successfully")
        
        return response
        
    except Exception as e:
        logger.exception("❌ Error in explain_dataset_directly")
        raise e

async def general_information(question, required_columns=None):
    """
    Handle general information queries using Tavily search
    
    Args:
        question (str): The user's question
        required_columns (list, optional): List of required column names
    """
    global CURRENT_CONVERSATION_ID
    
    try:

        print("Working for the general information")
        resp = client.serving_endpoints.query(
            SERVING_ENDPOINT_NAME,
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="""You are an expert assistant that provides accurate information based on the context provided.
                    Use the provided information to give a helpful, concise, and accurate response.
                    Please give me the answer in a beautiful readable format without * or | or any other special character
                    If the information doesn't contain the answer, say so instead of making up information."""
                ),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    #content=f"Based on the following information, please answer this question: {question}\n\nInformation: {content}"
                    content=f"Based on your best knowledge, please answer this question: {question}"
                ),
            ],
        )
        
        response = _extract_message_content(resp)
        logger.info(f"✅ Generated general information response for: {question}")
        
        return response

    except Exception as e:
        logger.exception("❌ Error in general_information")
        raise e

async def final_answer_combine(question, required_columns=None, ai_function_type=None, ai_function_types=None):
    """
    Process user question through Tavily search and Databricks Genie.
    Adjusts filtering and prompt based on keywords in the question.
    
    Args:
        question (str): The user's question
        required_columns (list, optional): List of required column names
        ai_function_type (str, optional): Primary AI function type from classification (for backward compatibility)
        ai_function_types (list, optional): List of all AI function types from classification
    """
    global CURRENT_CONVERSATION_ID, DYNAMIC_GENIE_ROOM_ID
    
    try:
        if not DYNAMIC_GENIE_ROOM_ID:
            raise Exception("Genie Room ID not set. Please configure it first.")
            
        lower_q = question.lower()

        contains_forecast = bool(re.search(r'\b(forecast|forecasted|forecasting)\b', lower_q))
        contains_classify = bool(re.search(r'\b(classify|classified|classification)\b', lower_q))

        if ai_function_types and len(ai_function_types) > 0:
            primary_function = ai_function_types[0]
            if primary_function == "ai_forecast":
                question = rephrase_query_forecast(question)
                logger.info(f"🔮 Rephrased forecast query: {question}")
                print(f"🔮 Rephrased forecast query: {question}")
                filter_keyword = primary_function
                logger.info(f"🎯 Using primary AI function type: {primary_function}")
            else:
                filter_keyword = primary_function
                logger.info(f"🎯 Using primary AI function type: {primary_function}")
        elif ai_function_type:
            if ai_function_type == "ai_forecast":
                question = rephrase_query_forecast(question)
                logger.info(f"🔮 Rephrased forecast query: {question}")
                print(f"🔮 Rephrased forecast query: {question}")
                filter_keyword = ai_function_type
                logger.info(f"🎯 Using AI function type: {ai_function_type}")
            else:
                filter_keyword = ai_function_type
                logger.info(f"🎯 Using AI function type: {ai_function_type}")
        elif contains_forecast:
            question = rephrase_query_forecast(question)
            logger.info(f"🔮 Rephrased forecast query: {question}")
            print(f"🔮 Rephrased forecast query: {question}")
            filter_keyword = "ai_forecast"
            logger.info("🔮 Detected forecast query, filtering for ai_forecast")
        elif contains_classify:
            filter_keyword = "ai_classify"
            logger.info("🏷️ Detected classify query, filtering for ai_classify")
        else:
            filter_keyword = "ai_query"
            logger.info("🧠 Processing standard predictive query, filtering for ai_query")

        logger.info(f"🔍 Looking for manual content for: {filter_keyword}")

        global MANUAL_AI_CONTENT
        combined_text = ""
        
        if ai_function_types and len(ai_function_types) > 1:
            logger.info(f"🔄 Processing multiple AI functions: {ai_function_types}")
            combined_contents = []
            for func_type in ai_function_types:
                content = MANUAL_AI_CONTENT.get(func_type, "")
                if content:
                    combined_contents.append(f"========{func_type.upper()}======\n{content}")
                else:
                    logger.warning(f"No manual content found for '{func_type}'")
            
            if combined_contents:
                combined_text = "\n\n".join(combined_contents)
                logger.info(f"✅ Combined manual content for {len(ai_function_types)} AI functions (total length: {len(combined_text)} characters)")
                print(f"✅ Combined manual content for {len(ai_function_types)} AI functions (total length: {len(combined_text)} characters)")
            else:
                raise Exception(f"No manual content configured for any of the AI function types: {ai_function_types}. Please set manual content first using /set-ai-content endpoint.")
        else:
            combined_text = MANUAL_AI_CONTENT.get(filter_keyword, "")
            
            if not combined_text:
                logger.warning(f"No manual content found for '{filter_keyword}'. Please set manual content using /set-ai-content endpoint.")
                raise Exception(f"No manual content configured for AI function type: {filter_keyword}. Please set manual content first using /set-ai-content endpoint.")
            
            combined_text = f"========{filter_keyword.upper()}======\n{combined_text}"
            logger.info(f"✅ Using manual content for '{filter_keyword}' (length: {len(combined_text)} characters)")
            print(f"✅ Using manual content for '{filter_keyword}' (length: {len(combined_text)} characters)")

        column_context = ""
        if required_columns and isinstance(required_columns, list) and len(required_columns) > 0:
            column_context = f"Focus on these columns and show the this columns in the output by creating new column for result: {', '.join(required_columns)}\n\n"

        task_description = {
            "ai_forecast": "forecast",
            "ai_classify": "classification logic",
            "ai_query": "SQL"
        }.get(filter_keyword, "SQL")

        if contains_forecast:
            forecast_prompt = (
                "Based on the following information, answer the following question:\n\n"
                f"{combined_text}\n\n"
                f"{column_context}Use the above information only just try to understand the question and write only the AI {task_description} function sql query for this question:\n"
                f"{question}\n"
            )
            logger.info("🚀 Sending forecast request to Genie...")
            logger.info(f"Using existing conversation ID: {CURRENT_CONVERSATION_ID}")
            
            response = await fetch_answer(fallback_workspace_url, DYNAMIC_GENIE_ROOM_ID, None, 
                                        forecast_prompt, CURRENT_CONVERSATION_ID)
        else:
            prompt = (
            "You are an expert SQL reasoning assistant. Your task is to read and understand the information below and generate an SQL function query strictly based on the logic and functions mentioned.\n\n"
            "=== INPUT DATA AND LOGIC ===\n"
            f"{combined_text}\n\n"
            "=== COLUMN CONTEXT ===\n"
            f"{column_context}\n\n"
            "=== TASK ===\n"
            f"{task_description}\n\n"
            "=== QUESTION ===\n"
            f"{question}\n\n"
            "=== INSTRUCTIONS ===\n"
            "- Use ONLY the logic and functions mentioned in the provided information.\n"
            "- If two or more functions are referenced, try to apply them all if relevant.\n"
            "- Do NOT make assumptions or introduce new logic.\n"
            "- Your output must be a single AI-generated SQL function query.\n"
            "- Unless explicitly referenced, apply the logic to only the first 10 rows of data\n"
            "- Output ONLY the SQL function query—no explanation, no markdown.\n"
        )

            logger.info("🚀 Sending request to Genie...")
            logger.info(f"Using existing conversation ID: {CURRENT_CONVERSATION_ID}")
            
            response = await fetch_answer(fallback_workspace_url, DYNAMIC_GENIE_ROOM_ID, None, 
                                        prompt, CURRENT_CONVERSATION_ID)
        
        if isinstance(response, dict) and "conversation_id" in response:
            CURRENT_CONVERSATION_ID = response["conversation_id"]
            logger.info(f"✅ Updated conversation ID: {CURRENT_CONVERSATION_ID}")

        return response

    except Exception as e:
        logger.exception("❌ Error in final_answer_combine")
        raise e

async def direct_genie_answer(question, required_columns=None):
    """
    Process user question directly with Databricks Genie without external search
    
    Args:
        question (str): The user's question
        required_columns (list, optional): List of required column names
    """
    global CURRENT_CONVERSATION_ID, DYNAMIC_GENIE_ROOM_ID
    
    try:
        if not DYNAMIC_GENIE_ROOM_ID:
            raise Exception("Genie Room ID not set. Please configure it first.")
            
        logger.info("🚀 Sending direct request to Genie...")
        logger.info(f"Using existing conversation ID: {CURRENT_CONVERSATION_ID}")
        
        question = question + " using the data."
        
        if required_columns and isinstance(required_columns, list) and len(required_columns) > 0:
            columns_str = ", ".join(required_columns)
            question = f"{question} Please focus on these columns : {columns_str}."
            logger.info(f"Enhanced question with column info: {question}")
        
        response = await fetch_answer(fallback_workspace_url, DYNAMIC_GENIE_ROOM_ID, None, 
                                     question, CURRENT_CONVERSATION_ID)
        
        if isinstance(response, dict) and "conversation_id" in response:
            CURRENT_CONVERSATION_ID = response["conversation_id"]
            logger.info(f"✅ Updated conversation ID: {CURRENT_CONVERSATION_ID}")
            
        return response
    except Exception as e:
        logger.exception("Error in direct_genie_answer")
        raise e


def extract_pdf_text(file_bytes: bytes, max_chars: int = 40000, password: str | None = None) -> str:
    """
    Extract text from a PDF. Handles encrypted PDFs and degrades gracefully.
    - If the PDF is encrypted and no/invalid password is provided, raises ValueError.
    - If `cryptography` is missing for AES-encrypted PDFs, raises RuntimeError with a clear message.
    """
    from io import BytesIO
    import re
    try:
        from pypdf import PdfReader
        from pypdf.errors import DependencyError as PdfDependencyError, PdfReadError
    except Exception:
        # If pypdf import itself fails, raise a clear error
        raise RuntimeError("pypdf is required to process PDFs. Please add 'pypdf' to requirements.txt.")
    try:
        bio = BytesIO(file_bytes)
        reader = PdfReader(bio)
        # Encrypted?
        if getattr(reader, "is_encrypted", False):
            try:
                # Try blank password first, or use provided
                result = reader.decrypt(password or "")
            except PdfDependencyError as e:
                # cryptography missing
                raise RuntimeError("Encrypted PDF requires 'cryptography>=3.1'. Please add it to requirements and rebuild.") from e
            # pypdf returns 0/False when wrong
            if not result:
                raise ValueError("PDF is encrypted. A valid password was not provided.")
        # Extract text page by page
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        text = "\n".join(texts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text[:max_chars]
    except PdfDependencyError as e:
        raise RuntimeError("Encrypted PDF requires 'cryptography>=3.1'. Please add it to requirements and rebuild.") from e
    except PdfReadError as e:
        raise ValueError(f"Invalid or corrupted PDF: {e}")


def build_prompt(pdf_content: str, question: str) -> str:
    """Build prompt for PDF-based questions"""
    return f"""
=== PDF CONTENT ===
{pdf_content}

=== USER'S ORIGINAL QUESTION ===
{question}
""".strip()


def ask_databricks(pdf_content: str, question: str) -> str:
    """Ask Databricks serving endpoint with PDF context"""
    prompt = build_prompt(pdf_content, question)
    resp = client.serving_endpoints.query(
        SERVING_ENDPOINT_NAME,
        messages=[
            ChatMessage(
                role=ChatMessageRole.SYSTEM,
                content=("You are an expert in answering questions based on PDF content. "
                         "Give the answer in less than 100 words. Give me a beautiful summary.")
            ),
            ChatMessage(role=ChatMessageRole.USER, content=prompt),
        ],
    )
    try:
        return _extract_message_content(resp)
    except Exception:
        return "I couldn't produce an answer from the endpoint."


class ChatIn(BaseModel):
    """Model for chat input with optional session ID for PDF mode"""
    question: str
    session_id: Optional[str] = None  # when provided, force QA over stored PDF



@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    from fastapi.responses import JSONResponse
    from fastapi import UploadFile, File, HTTPException
    import uuid
    """Upload and process PDF file (returns a session_id). Always responds with JSON."""
    if file.content_type not in ("application/pdf",):
        raise HTTPException(status_code=400, detail="Please upload a PDF.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        pdf_content = extract_pdf_text(data)
    except ValueError as e:
        # e.g., encrypted without password, invalid PDF, etc.
        return JSONResponse(status_code=400, content={"error": str(e)})
    except RuntimeError as e:
        # e.g., missing cryptography for AES
        return JSONResponse(status_code=500, content={"error": str(e)})
    except Exception as e:
        # Catch-all to ensure frontend always gets JSON
        return JSONResponse(status_code=500, content={"error": f"Failed to process PDF: {str(e)}"})

    session_id = str(uuid.uuid4())
    # Assumes SESSION_STORE exists; keep behavior unchanged
    try:
        SESSION_STORE[session_id] = pdf_content
    except Exception:
        # Fallback if SESSION_STORE isn't defined for some reason
        pass
    return JSONResponse({"session_id": session_id})


@app.post("/chat")
def chat(body: ChatIn):
    """Handle chat with optional PDF mode"""
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # If session_id present & valid → ALWAYS answer using PDF (QA with PDF)
    if body.session_id:
        pdf_content = SESSION_STORE.get(body.session_id)
        if not pdf_content:
            raise HTTPException(status_code=400, detail="Invalid or expired session_id. Upload the PDF again.")
        answer = ask_databricks(pdf_content=pdf_content, question=question)
        return JSONResponse({"answer": answer})

    return JSONResponse({"answer": "PDF mode is off. Turn on the checkbox (and upload a PDF) to answer from the document."})


@app.post("/set-genie-id")
async def set_genie_id(request: Request):
    """Set the Genie Room ID configuration"""
    global DYNAMIC_GENIE_ROOM_ID, CURRENT_CONVERSATION_ID
    
    try:
        body = await request.json()
        genie_room_id = body.get("genie_room_id", "").strip()
        
        if not genie_room_id:
            return JSONResponse({"success": False, "error": "Genie Room ID is required"}, status_code=400)
        
        DYNAMIC_GENIE_ROOM_ID = genie_room_id
        
        CURRENT_CONVERSATION_ID = None
        
        logger.info(f"✅ Genie Room ID updated: {DYNAMIC_GENIE_ROOM_ID}")
        
        return JSONResponse({
            "success": True, 
            "message": "Genie Room ID set successfully",
            "genie_room_id": genie_room_id
        })
        
    except Exception as e:
        logger.exception("Error setting configuration")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/set-ai-content")
async def set_ai_content(request: Request):
    """Set manual content for specific AI function types"""
    global MANUAL_AI_CONTENT
    
    try:
        body = await request.json()
        ai_function_type = body.get("ai_function_type", "").strip()
        content = body.get("content", "").strip()
        
        if not ai_function_type:
            return JSONResponse({"success": False, "error": "AI function type is required"}, status_code=400)
        
        if not content:
            return JSONResponse({"success": False, "error": "Content is required"}, status_code=400)
        
        if ai_function_type not in MANUAL_AI_CONTENT:
            return JSONResponse({
                "success": False, 
                "error": f"Invalid AI function type. Valid types are: {list(MANUAL_AI_CONTENT.keys())}"
            }, status_code=400)
        
        MANUAL_AI_CONTENT[ai_function_type] = content
        
        logger.info(f"✅ Manual content set for {ai_function_type}")
        
        return JSONResponse({
            "success": True, 
            "message": f"Manual content set successfully for {ai_function_type}",
            "ai_function_type": ai_function_type,
            "content_length": len(content)
        })
        
    except Exception as e:
        logger.exception("Error setting manual AI content")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/get-ai-content")
async def get_ai_content():
    """Get all manual content for AI function types"""
    global MANUAL_AI_CONTENT
    
    try:
        content_info = {}
        for ai_type, content in MANUAL_AI_CONTENT.items():
            content_info[ai_type] = {
                "content": content,
                "has_content": bool(content.strip()),
                "content_length": len(content)
            }
        
        return JSONResponse({
            "success": True,
            "ai_content": content_info
        })
        
    except Exception as e:
        logger.exception("Error getting manual AI content")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/get-ai-content/{ai_function_type}")
async def get_specific_ai_content(ai_function_type: str):
    """Get manual content for a specific AI function type"""
    global MANUAL_AI_CONTENT
    
    try:
        if ai_function_type not in MANUAL_AI_CONTENT:
            return JSONResponse({
                "success": False, 
                "error": f"Invalid AI function type. Valid types are: {list(MANUAL_AI_CONTENT.keys())}"
            }, status_code=400)
        
        content = MANUAL_AI_CONTENT[ai_function_type]
        
        return JSONResponse({
            "success": True,
            "ai_function_type": ai_function_type,
            "content": content,
            "has_content": bool(content.strip()),
            "content_length": len(content)
        })
        
    except Exception as e:
        logger.exception("Error getting specific AI content")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/set-ai-content-bulk")
async def set_ai_content_bulk(request: Request):
    """Set manual content for multiple AI function types at once"""
    global MANUAL_AI_CONTENT
    
    try:
        body = await request.json()
        ai_content_dict = body.get("ai_content", {})
        
        if not ai_content_dict or not isinstance(ai_content_dict, dict):
            return JSONResponse({
                "success": False, 
                "error": "ai_content must be a dictionary with AI function types as keys and content as values"
            }, status_code=400)
        
        invalid_types = []
        for ai_type in ai_content_dict.keys():
            if ai_type not in MANUAL_AI_CONTENT:
                invalid_types.append(ai_type)
        
        if invalid_types:
            return JSONResponse({
                "success": False, 
                "error": f"Invalid AI function types: {invalid_types}. Valid types are: {list(MANUAL_AI_CONTENT.keys())}"
            }, status_code=400)
        
        updated_types = []
        for ai_type, content in ai_content_dict.items():
            if content and content.strip():
                MANUAL_AI_CONTENT[ai_type] = content.strip()
                updated_types.append(ai_type)
        
        logger.info(f"✅ Bulk manual content set for: {updated_types}")
        
        return JSONResponse({
            "success": True, 
            "message": f"Manual content set successfully for {len(updated_types)} AI function types",
            "updated_types": updated_types
        })
        
    except Exception as e:
        logger.exception("Error setting bulk manual AI content")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/delete-ai-content/{ai_function_type}")
async def delete_ai_content(ai_function_type: str):
    """Delete manual content for a specific AI function type"""
    global MANUAL_AI_CONTENT
    
    try:
        if ai_function_type not in MANUAL_AI_CONTENT:
            return JSONResponse({
                "success": False, 
                "error": f"Invalid AI function type. Valid types are: {list(MANUAL_AI_CONTENT.keys())}"
            }, status_code=400)
        
        MANUAL_AI_CONTENT[ai_function_type] = ""
        
        logger.info(f"✅ Manual content deleted for {ai_function_type}")
        
        return JSONResponse({
            "success": True,
            "message": f"Manual content deleted successfully for {ai_function_type}",
            "ai_function_type": ai_function_type
        })
        
    except Exception as e:
        logger.exception("Error deleting AI content")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/delete-ai-content-all")
async def delete_all_ai_content():
    """Delete all manual content for AI function types"""
    global MANUAL_AI_CONTENT
    
    try:
        for ai_type in MANUAL_AI_CONTENT.keys():
            MANUAL_AI_CONTENT[ai_type] = ""
        
        logger.info("✅ All manual content deleted")
        
        return JSONResponse({
            "success": True,
            "message": "All manual content deleted successfully",
            "cleared_types": list(MANUAL_AI_CONTENT.keys())
        })
        
    except Exception as e:
        logger.exception("Error deleting all AI content")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/create-tracking-table")
async def create_tracking_table_endpoint(request: Request):
    """Create the user interaction tracking table"""
    try:
        if not fallback_workspace_url:
            return JSONResponse({
                "success": False,
                "error": "Workspace URL not configured in environment. Please check .env file."
            }, status_code=400)
        if not fallback_access_token:
            return JSONResponse({
                "success": False,
                "error": "Access Token not configured in environment. Please check .env file."
            }, status_code=400)
            
        body = await request.json()
        catalog_name = body.get("catalog_name", "users_trial")
        schema_name = body.get("schema_name", "nitin_aggarwal")
        
        result = create_user_interaction_table(
            catalog_name=catalog_name,
            schema_name=schema_name,
            workspace_url=fallback_workspace_url,
            access_token=fallback_access_token
        )
        
        if result["success"]:
            logger.info(f"✅ Tracking table created in {catalog_name}.{schema_name}")
            return JSONResponse({
                "success": True,
                "message": f"Tracking table created successfully in {catalog_name}.{schema_name}",
                "catalog_name": catalog_name,
                "schema_name": schema_name
            })
        else:
            return JSONResponse({
                "success": False,
                "error": result["error"]
            }, status_code=400)
            
    except Exception as e:
        logger.exception("Error creating tracking table")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# --- New: simple health/probe endpoint to avoid 404 spam ---
@app.get("/stats")
async def stats():
    return {"ok": True}

@app.get("/")
async def home(request: Request):
    """Render the home page with chat interface"""
    global CURRENT_CONVERSATION_ID, DYNAMIC_GENIE_ROOM_ID
    CURRENT_CONVERSATION_ID = None  # Reset the conversation when the home page is loaded
    logger.info("🔄 Conversation ID reset on page load")
    
    if DYNAMIC_GENIE_ROOM_ID:
        logger.info(f"✅ Genie Room ID loaded from environment: {DYNAMIC_GENIE_ROOM_ID}")
    else:
        logger.warning("⚠️ No Genie Room ID found in environment variable GENIE_ROOM_ID")
    
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/query")
async def query(request: Request): 
    """Process user query and return response"""
    try:
        if not fallback_workspace_url:
            return JSONResponse({
                "error": "Workspace URL not configured in environment. Please check .env file."
            }, status_code=400)
        if not fallback_access_token:
            return JSONResponse({
                "error": "Access Token not configured in environment. Please check .env file."
            }, status_code=400)
        if not DYNAMIC_GENIE_ROOM_ID:
            return JSONResponse({
                "error": "Genie Room ID not configured. Please set it first."
            }, status_code=400)
            
        body = await request.json()
        user_query = body.get("query", "").strip()
        
        # Check for PDF mode
        session_id = body.get("session_id")
        if session_id:
            pdf_content = SESSION_STORE.get(session_id)
            if not pdf_content:
                return JSONResponse({
                    "error": "Invalid or expired session_id. Upload the PDF again."
                }, status_code=400)
            
            
            try:
                answer = ask_databricks(pdf_content=pdf_content, question=user_query)
                return JSONResponse({
                    "response_type": "text",
                    "message": answer,
                    "original_query": user_query,
                    "query_classification": "pdf_qa",
                    "ai_function_type": "pdf_chat"
                })
            except Exception as e:
                logger.exception("Error in PDF Q&A")
                return JSONResponse({
                    "error": f"Error processing PDF question: {str(e)}"
                }, status_code=500)
        
        catalog_name = body.get("catalog_name")
        schema_name = body.get("schema_name")
        
        reset_conversation = body.get("reset_conversation", False)
        
        if reset_conversation:
            global CURRENT_CONVERSATION_ID
            CURRENT_CONVERSATION_ID = None
            logger.info("🔄 Conversation reset requested - starting new conversation")
            return JSONResponse({
                "response_type": "text",
                "message": "Conversation has been reset. Starting a new conversation."
            })
        
        if not user_query:
            return JSONResponse({"error": "Query parameter is required"}, status_code=400)
        
        if catalog_name and schema_name:
            context_message = f"Using catalog '{catalog_name}' and schema '{schema_name}': "
            user_query_with_context = f"{context_message}{user_query}"
            logger.info(f"Query with catalog/schema context: {user_query_with_context}")
        else:
            user_query_with_context = user_query
        
        query_for_classification = user_query_with_context
        query_for_processing = user_query_with_context
        
        table_schema = {}
        if catalog_name and schema_name:
            try:
                tables_result = get_tables(catalog_name, schema_name, fallback_workspace_url, None)
                if tables_result["success"]:
                    table_schema = {}
                    for table in tables_result["tables"]:
                        table_name = table["name"]
                        try:
                            columns_result = get_table_columns(catalog_name, schema_name, table_name, fallback_workspace_url, None)
                            
                            if columns_result["success"]:
                                columns = columns_result["columns"]
                                table_schema[table_name] = columns
                                logger.info(f"Retrieved columns for {table_name}: {columns}")
                            else:
                                logger.warning(f"API call failed for {table_name}: {columns_result.get('error')}")
                                
                                describe_query = f"DESCRIBE TABLE {catalog_name}.{schema_name}.{table_name}"
                                describe_response = await fetch_answer(fallback_workspace_url, DYNAMIC_GENIE_ROOM_ID, None, 
                                                                    describe_query, None)
                                
                                columns = []
                                if isinstance(describe_response, dict) and "statement_response" in describe_response:
                                    stmt = describe_response["statement_response"]
                                    if stmt and "result" in stmt and "data_array" in stmt["result"]:
                                        columns = [row[0] for row in stmt["result"]["data_array"] if row]
                                
                                table_schema[table_name] = columns
                                logger.info(f"Retrieved columns for {table_name} using fallback: {columns}")
                        except Exception as col_err:
                            logger.warning(f"Failed to get columns for table {table_name}: {str(col_err)}")
                            table_schema[table_name] = []
            except Exception as e:
                logger.warning(f"Failed to get table schema: {str(e)}")
        
        dataset_phrases = [
            "explain the dataset",
            "what does this dataset represent",
            "describe the dataset",
            "give an overview of the dataset",
            "summarize the dataset",
            "tell me about the dataset",
            "what is this dataset about",
            "dataset explanation",
            "dataset summary",
            "overview of the dataset",
            "what's in the dataset",
            "explain the data",
            "describe the data",
            "what does this data represent"
        ]
        
        user_query_lower = user_query_with_context.lower()
        if any(phrase in user_query_lower for phrase in dataset_phrases):
            logger.info("🎯 Detected 'explain the dataset' query - bypassing classification")
            try:
                response = await explain_dataset_directly(table_schema, user_query)
                
                try:
                    log_result = log_user_interaction(
                        user_question=user_query,
                        genie_space_id=DYNAMIC_GENIE_ROOM_ID,
                        ai_function_type="direct_explanation",
                        query_classification="dataset_explanation",
                        conversation_id=None,
                        required_columns=[],
                        response_type="text",
                        user_email=None,
                        workspace_url=fallback_workspace_url,
                        access_token=fallback_access_token
                    )
                    
                    if log_result and log_result.get("success"):
                        logger.info("✅ Dataset explanation interaction logged")
                        
                except Exception as tracking_error:
                    logger.warning(f"⚠️ Failed to log dataset explanation interaction: {str(tracking_error)}")
                
                return JSONResponse({
                    "response_type": "text",
                    "message": response,
                    "original_query": user_query,
                    "conversation_id": None,
                    "catalog_name": catalog_name,
                    "schema_name": schema_name,
                    "required_columns": [],
                    "query_classification": "dataset_explanation",
                    "ai_function_type": "direct_explanation"
                })
                
            except Exception as e:
                logger.exception("❌ Error in dataset explanation")
                return JSONResponse({
                    "response_type": "text",
                    "message": f"❌ Error explaining dataset: {str(e)}"
                }, status_code=500)
        
        query_type_result = classify_query(query_for_classification)
        logger.info(f"🧠 Query classified as: {query_type_result}")
        
        ai_function_types = []
        ai_function_type = None
        
        if "predictive sql" in query_type_result.lower():
            ai_function_types = classify_predictive_query(query_for_classification)
            logger.info(f"🎯 Predictive query classified as AI functions: {ai_function_types}")
            ai_function_type = ai_function_types[0] if ai_function_types else None
        
        logger.info(f"Table schema being passed: {table_schema}")
        required_columns = determine_required_columns(query_for_processing, table_schema)
        logger.info(f"Required columns determined: {required_columns}")
        
        classification = query_type_result.lower()
        
        if "normal sql" in classification:
            response = await direct_genie_answer(query_for_processing, required_columns)
        elif "predictive sql" in classification:
            response = await final_answer_combine(query_for_processing, required_columns, ai_function_type, ai_function_types)
        elif "general information" in classification:
            response = await general_information(query_for_processing, required_columns)
        else:
            return JSONResponse({
                "response_type": "text",
                "message": f"❌ Unrecognized query classification: {query_type_result}"
            }, status_code=400)
        
        logger.info(f"✅ Raw response received: {response}")
        
        if isinstance(response, dict) and "conversation_id" in response:
            conversation_id = response["conversation_id"]
        else:
            conversation_id = CURRENT_CONVERSATION_ID
        
        try:
            primary_ai_function = ai_function_types[0] if ai_function_types and len(ai_function_types) > 0 else ai_function_type
            
            log_result = log_user_interaction(
                user_question=user_query,
                genie_space_id=DYNAMIC_GENIE_ROOM_ID,
                ai_function_type=primary_ai_function,
                query_classification=query_type_result,
                conversation_id=conversation_id,
                required_columns=required_columns,
                response_type="table" if isinstance(response, dict) and "statement_response" in response else "text",
                user_email=None,
                workspace_url=fallback_workspace_url,
                access_token=fallback_access_token
            )
            
            if log_result and log_result.get("success"):
                logger.info("✅ Interaction logged")
                
        except Exception as tracking_error:
            logger.warning(f"⚠️ Failed to log user interaction: {str(tracking_error)}")
        
        contains_forecast = bool(re.search(r'\b(forecast|forecasted|forecasting)\b', user_query.lower()))
        
        try:
            stmt = response.get("statement_response") if isinstance(response, dict) else None
            if stmt:
                manifest = stmt.get("manifest", {})
                total_row_count = manifest.get("total_row_count", 0)
                
                if total_row_count == 0:
                    sql_query = response.get("sql_query") if isinstance(response, dict) else None
                    
                    return JSONResponse({
                        "response_type": "text",
                        "message": "There is no data related to query",
                        "sql_query": sql_query,
                        "original_query": user_query,
                        "rephrased_query": None,
                        "conversation_id": conversation_id,
                        "catalog_name": catalog_name,
                        "schema_name": schema_name,
                        "required_columns": required_columns,
                        "query_classification": query_type_result,
                        "ai_function_type": ai_function_type,
                        "ai_function_types": ai_function_types
                    })
                
                columns_info = manifest.get("schema", {}).get("columns")
                data_array = stmt.get("result", {}).get("data_array")

                if columns_info and data_array:
                    columns = [col["name"] for col in columns_info]
                    df = pd.DataFrame(data_array, columns=columns)
                    
                    sql_query = response.get("sql_query") if isinstance(response, dict) else None
                    logger.info(f"🔍 SQL query extracted for table response: {sql_query[:100] if sql_query else 'None'}")
                    
                    response_data = {
                        "response_type": "table",
                        "columns": df.columns.tolist(),
                        "data": df.to_dict(orient="records"),
                        "sql_query": sql_query,
                        "original_query": user_query,
                        "rephrased_query": None,
                        "conversation_id": conversation_id,
                        "catalog_name": catalog_name,
                        "schema_name": schema_name,
                        "required_columns": required_columns,
                        "query_classification": query_type_result,
                        "ai_function_type": ai_function_type,
                        "ai_function_types": ai_function_types
                    }
                    
                    logger.info(f"🔍 Final response data includes sql_query: {'sql_query' in response_data and response_data['sql_query'] is not None}")
                    return JSONResponse(response_data)

            if isinstance(response, dict) and "answer" in response:
                sql_query = response.get("sql_query")
                
                return JSONResponse({
                    "response_type": "text",
                    "message": response["answer"],
                    "sql_query": sql_query,
                    "original_query": user_query,
                    "rephrased_query": None,
                    "conversation_id": conversation_id,
                    "catalog_name": catalog_name,
                    "schema_name": schema_name,
                    "required_columns": required_columns,
                    "query_classification": query_type_result,
                    "ai_function_type": ai_function_type
                })

            sql_query = response.get("sql_query") if isinstance(response, dict) else None
            
            return JSONResponse({
                "response_type": "text",
                "message": str(response),
                "sql_query": sql_query,
                "original_query": user_query,
                "rephrased_query": None,
                "conversation_id": conversation_id,
                "catalog_name": catalog_name,
                "schema_name": schema_name,
                "required_columns": required_columns,
                "query_classification": query_type_result,
                "ai_function_type": ai_function_type
            })
        
        except Exception as parse_err:
            logger.exception("⚠️ Failed to parse response structure")
            
            sql_query = response.get("sql_query") if isinstance(response, dict) else None
            
            return JSONResponse({
                "response_type": "text",
                "message": "⚠️ Unexpected response format.",
                "sql_query": sql_query,
                "raw_response": str(response),
                "original_query": user_query,
                "rephrased_query": None,
                "conversation_id": conversation_id,
                "catalog_name": catalog_name,
                "schema_name": schema_name,
                "required_columns": required_columns,
                "query_classification": query_type_result,
                "ai_function_type": ai_function_type
            })

    except Exception as e:
        logger.exception("🔥 Error processing query")
        return JSONResponse({
            "response_type": "text", 
            "message": f"❌ Error: {str(e)}"
        }, status_code=500)


@app.get("/reset-conversation")
async def reset_conversation():
    """Endpoint to explicitly reset the conversation"""
    global CURRENT_CONVERSATION_ID
    CURRENT_CONVERSATION_ID = None
    logger.info("🔄 Conversation has been reset")
    return JSONResponse({"status": "success", "message": "Conversation reset successful"})


@app.get("/conversation-status")
async def conversation_status():
    """Return the current conversation ID status"""
    global CURRENT_CONVERSATION_ID
    return JSONResponse({
        "has_active_conversation": CURRENT_CONVERSATION_ID is not None,
        "conversation_id": CURRENT_CONVERSATION_ID
    })

@app.get("/genie-config-status")
async def genie_config_status():
    """Return the current genie configuration status"""
    global DYNAMIC_GENIE_ROOM_ID
    return JSONResponse({
        "genie_configured": DYNAMIC_GENIE_ROOM_ID is not None,
        "genie_room_id": DYNAMIC_GENIE_ROOM_ID,
        "source": "environment_variable" if DYNAMIC_GENIE_ROOM_ID == GENIE_ROOM_ID_FROM_ENV else "frontend_override"
    })


@app.get("/sample-questions")
async def sample_questions(section: str):
    """Return sample questions for the user interface based on section"""
    global SCHEMA_INFO
    
    questions = generate_dynamic_questions(SCHEMA_INFO)
    return JSONResponse({"questions": questions})

@app.get("/catalogs")
async def get_catalogs():
    """Return a list of available catalogs"""
    catalogs = [
        {"name": "users_trial", "display_name": "Users Trial"},
        {"name": "main", "display_name": "Main"},
        {"name": "hive_metastore", "display_name": "Hive Metastore"}
    ]
    return JSONResponse({"catalogs": catalogs})

@app.get("/schemas/{catalog_name}")
async def get_schemas(catalog_name: str):
    """Return schemas for a given catalog"""
    schemas = []
    if catalog_name == "users_trial":
        schemas = [
            {"name": "nitin_aggarwal", "display_name": "Nitin Aggarwal"},
            {"name": "default", "display_name": "Default"}
        ]
    elif catalog_name == "main":
        schemas = [
            {"name": "default", "display_name": "Default"},
            {"name": "samples", "display_name": "Samples"}
        ]
    elif catalog_name == "hive_metastore":
        schemas = [
            {"name": "default", "display_name": "Default"}
        ]
    return JSONResponse({"schemas": schemas})

@app.get("/tables/{catalog_name}/{schema_name}")
async def get_catalog_tables(catalog_name: str, schema_name: str):
    """Return tables for a given catalog and schema"""
    try:
        result = get_tables(catalog_name, schema_name, fallback_workspace_url, None)
        if result["success"]:
            formatted_tables = []
            for table in result["tables"]:
                formatted_tables.append({
                    "name": table.get("name"),
                    "full_name": table.get("full_name"),
                    "table_type": table.get("table_type")
                })
            return JSONResponse({"tables": formatted_tables})
        else:
            return JSONResponse({"error": result["error"]}, status_code=400)
    except Exception as e:
        logger.exception("Error fetching tables")
        return JSONResponse({"error": str(e)}, status_code=500)



@app.post("/fetch-schema")
async def fetch_schema(request: Request):
    """Fetch schema information by getting first 5 rows of data"""
    global DYNAMIC_GENIE_ROOM_ID, SCHEMA_INFO
    
    try:
        body = await request.json()
        genie_room_id = body.get("genie_room_id", "").strip()
        
        if not genie_room_id:
            return JSONResponse({"success": False, "error": "Genie Room ID is required"}, status_code=400)
        
        if not fallback_workspace_url:
            return JSONResponse({"success": False, "error": "Workspace URL not configured in environment"}, status_code=400)
        if not fallback_access_token:
            return JSONResponse({"success": False, "error": "Access Token not configured in environment"}, status_code=400)
        
        DYNAMIC_GENIE_ROOM_ID = genie_room_id
        logger.info(f"✅ Using Genie Room ID: {DYNAMIC_GENIE_ROOM_ID}")
        
        question = "Give me first 1 row and all columns from all the tables in the dataset using joins"
        
        logger.info("🚀 Fetching first 5 rows to extract schema...")
        
        response = await fetch_answer(fallback_workspace_url, genie_room_id, fallback_access_token, question, None)
        
        print("================schema infor", response)
        
        columns = []
        try:
            if isinstance(response, dict) and "statement_response" in response:
                stmt = response["statement_response"]
                if stmt:
                    columns_info = stmt.get("manifest", {}).get("schema", {}).get("columns")
                    if columns_info:
                        columns = [col["name"] for col in columns_info]
                        logger.info(f"✅ Extracted columns from manifest: {columns}")
                    
                    elif "result" in stmt and "data_array" in stmt["result"]:
                        data_array = stmt["result"]["data_array"]
                        if data_array and len(data_array) > 0:
                            first_row = data_array[0]
                            if first_row:
                                columns = [f"column_{i+1}" for i in range(len(first_row))]
                                logger.info(f"✅ Inferred columns from data structure: {columns}")
                    
                    elif "manifest" in stmt:
                        manifest = stmt["manifest"]
                        print("================manifest structure", manifest)
                        if "schema" in manifest:
                            schema = manifest["schema"]
                            print("================schema structure", schema)
        
        except Exception as e:
            logger.error(f"Error extracting columns: {str(e)}")
            print("================Error extracting columns:", str(e))
        
        if columns:
            SCHEMA_INFO = "\t".join(columns)
            logger.info(f"✅ Updated global SCHEMA_INFO: {SCHEMA_INFO}")
            
            return JSONResponse({
                "success": True,
                "message": "Schema information fetched successfully",
                "columns": columns,
                "schema_info": SCHEMA_INFO,
                "genie_room_id": genie_room_id
            })
        else:
            return JSONResponse({
                "success": False, 
                "error": "Could not extract column information from the response",
                "raw_response": str(response)
            }, status_code=400)
            
    except Exception as e:
        logger.exception("Error fetching schema information")
        return JSONResponse({
            "success": False, 
            "error": f"Failed to fetch schema: {str(e)}"
        }, status_code=500)

if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000)   #for databricks
    uvicorn.run(app, port=8000, timeout_keep_alive=600)
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

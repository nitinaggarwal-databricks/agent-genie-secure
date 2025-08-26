# 🏥 Databricks Healthcare Analytics Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?style=flat&logo=databricks&logoColor=white)](https://databricks.com/)
[![AI/ML](https://img.shields.io/badge/AI/ML-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://ai.databricks.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://python.org/)
[![Healthcare](https://img.shields.io/badge/Healthcare-2E8B57?style=flat&logo=hospital-symbol&logoColor=white)](#)

> **A sophisticated AI-powered healthcare analytics platform that combines Databricks' advanced AI capabilities with intuitive query processing for comprehensive healthcare data insights.**

## 🌟 Overview

The Databricks Healthcare Analytics Platform is an enterprise-grade solution that revolutionizes healthcare data analytics through intelligent query processing, multi-modal AI functions, and seamless integration with Databricks' ecosystem. Built for healthcare organizations seeking advanced analytics capabilities with enterprise security and scalability.

### 🎯 **Core Innovation: Intelligent Query Processing Pipeline**

Our platform features a groundbreaking 4-stage query processing system that automatically classifies, routes, and optimizes healthcare data queries:

```
📥 User Query → 🧠 AI Classification → 🔍 Column Analysis → 🚀 Intelligent Routing → 📊 Optimized Results
```

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    🌐 Frontend Interface                        │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │   Chat UI       │   PDF Upload    │   Table Visualization   │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                   🔧 FastAPI Backend                            │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │ Query Processor │ PDF Handler     │   User Tracking         │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│             🧠 AI Processing Layer                              │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │Query Classifier │ Column Analyzer │  AI Function Router     │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│              📊 Databricks Integration                          │
│  ┌─────────────────┬─────────────────┬─────────────────────────┐ │
│  │  Genie API      │ Unity Catalog   │   Serving Endpoints     │ │
│  └─────────────────┴─────────────────┴─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 **Query Processing Workflow - The Heart of the Platform**

### **Stage 1: Intelligent Query Classification** 🧠

Every incoming query is analyzed by our AI classification system:

```python
📝 Input: "Forecast patient readmissions for next quarter"
    ↓
🧠 AI Classifier Analysis:
    ├─ ❌ Normal SQL (standard database operations)
    ├─ ✅ Predictive SQL (requires AI/ML functions)  
    └─ ❌ General Information (knowledge queries)
    ↓
🏷️ Classification: "Predictive SQL"
```

**Query Types:**
- **Normal SQL**: Standard database queries (SELECT, JOIN, aggregations)
- **Predictive SQL**: AI-enhanced queries requiring machine learning
- **General Information**: Knowledge-based questions about healthcare concepts

### **Stage 2: Dynamic Column Analysis** 🔍

Our AI-powered column analyzer maps queries to database schema:

```python
📝 Query: "Show diabetes patients with high risk scores"
    ↓
🔍 Column Analysis:
    ├─ Required Columns Identified:
    │   ├─ patient_id (patient identification)
    │   ├─ diagnosis_code (condition filtering)
    │   ├─ diagnosis_description (diabetes detection)
    │   └─ risk_score (risk assessment)
    ↓
📋 Output: ["patient_id", "diagnosis_code", "diagnosis_description", "risk_score"]
```

### **Stage 3: AI Function Detection** 🎯

For predictive queries, our system identifies required AI functions:

```python
📝 Query: "Classify patient risk and forecast admissions"
    ↓
🎯 AI Function Detection:
    ├─ Primary Function: ai_classify (risk categorization)
    ├─ Secondary Function: ai_forecast (admission prediction)
    └─ Processing Mode: Multi-function pipeline
    ↓
🔧 Functions: ["ai_classify", "ai_forecast"]
```

### **Stage 4: Intelligent Response Generation** 🚀

Queries are routed to optimal processing engines:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Normal SQL    │    │ Predictive SQL  │    │General Knowledge│
│        ↓        │    │        ↓        │    │        ↓        │
│  Direct Genie   │    │  AI + Genie     │    │ Serving Endpoint│
│   API Query     │    │  Integration    │    │   Knowledge     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🎯 **Healthcare Use Cases & Examples**

### **Clinical Analytics** 🏥
```sql
-- Predictive Risk Assessment
"Classify patients into high, medium, low risk categories based on recent lab results"

-- Outcome Forecasting  
"Forecast ICU bed utilization for the next 30 days"

-- Anomaly Detection
"Identify unusual patterns in patient vital signs"
```

### **Operational Intelligence** 📊
```sql
-- Resource Planning
"Predict staffing needs for emergency department next week"

-- Quality Metrics
"Analyze patient satisfaction trends and predict improvement opportunities"

-- Cost Optimization
"Forecast medication costs and identify cost-saving opportunities"
```

### **Advanced AI Applications** 🤖
```sql
-- Natural Language Processing
"Summarize patient histories for discharge planning"
"Translate treatment plans into Spanish for Spanish-speaking patients"

-- Text Analysis
"Extract medication names and dosages from clinical notes"
"Analyze sentiment of patient feedback surveys"
```

---

## 🛠️ **Installation & Setup**

### **Prerequisites**
- Python 3.8+
- Databricks account with Unity Catalog access
- Genie Room configured in Databricks

### **1. Environment Setup**
```bash
# Clone the repository
git clone <repository-url>
cd nitin_app

# Create virtual environment
python -m venv nitin_venv
source nitin_venv/bin/activate  # On Windows: nitin_venv\Scripts\activate

# Install dependencies
cd ""
pip install -r requirements.txt
```

### **2. Configuration**
Environment variables are configured in the `""/app.yaml` file:

```yaml
env:
  - name: "SPACE_ID"
    value: "your_genie_room_id_here"
  - name: "SERVING_ENDPOINT_NAME"
    value: "your_serving_endpoint_name"
```

**Optional environment variables** can also be added to `app.yaml`:
```yaml
env:
  - name: "SPACE_ID"
    value: "your_genie_room_id_here"
  - name: "SERVING_ENDPOINT_NAME"
    value: "your_serving_endpoint_name"
  - name: "WORKSPACE_URL"
    value: "https://your-workspace.cloud.databricks.com"
  - name: "ACCESS_TOKEN"
    value: "your_access_token"
  - name: "ANSWER_CACHE_TTL"
    value: "300"
  - name: "ANSWER_CACHE_MAX"
    value: "512"
  - name: "ALLOW_INSECURE_SSL"
    value: "false"
```

### **3. Databricks Setup**
1. **Create Genie Room**: Set up a Genie Room in your Databricks workspace
2. **Configure Unity Catalog**: Ensure your healthcare data is accessible
3. **Set up Serving Endpoint**: Configure model serving for AI functions

### **4. Start the Application**
```bash

python app.py
```

Access the platform at: `http://localhost:8000`

---

## 📚 **API Reference**

### **Core Query Processing**

#### **POST /query**
Process healthcare data queries with intelligent routing.

**Request:**
```json
{
    "query": "Show me patients with diabetes admitted in the last 30 days",
    "catalog_name": "healthcare_data",
    "schema_name": "patient_records"
}
```

**Response:**
```json
{
    "response_type": "table",
    "columns": ["patient_id", "first_name", "last_name", "admit_date"],
    "data": [...],
    "sql_query": "SELECT patient_id, first_name, last_name, admit_date FROM...",
    "query_classification": "Normal SQL",
    "required_columns": ["patient_id", "diagnosis_code", "admit_datetime"]
}
```

### **Configuration Management**

#### **POST /set-genie-id**
Configure Genie Room ID for database access.

**Request:**
```json
{
    "genie_room_id": "your_room_id_here"
}
```

#### **POST /set-ai-content**
Configure AI function definitions for predictive queries.

**Request:**
```json
{
    "ai_function_type": "ai_forecast",
    "content": "Databricks AI forecast function syntax and examples..."
}
```

### **Document Processing**

#### **POST /upload_pdf**
Upload and process healthcare documents for Q&A.

**Request:** Multipart form with PDF file (max 3MB)

**Response:**
```json
{
    "session_id": "uuid-session-identifier"
}
```

#### **POST /chat**
Ask questions about uploaded documents.

**Request:**
```json
{
    "question": "What are the patient's current medications?",
    "session_id": "uuid-session-identifier"
}
```

### **Analytics & Tracking**

#### **POST /create-tracking-table**
Set up user interaction tracking for analytics.

#### **GET /sample-questions**
Get dynamically generated questions based on available data schema.

#### **GET /catalogs, /schemas/{catalog}, /tables/{catalog}/{schema}**
Browse available healthcare data catalogs and schemas.

---

## 🤖 **AI Functions & Capabilities**

### **Supported AI Functions**

| Function | Healthcare Use Case | Example |
|----------|-------------------|---------|
| `ai_classify` | Risk stratification, condition categorization | "Classify patients by readmission risk" |
| `ai_forecast` | Demand planning, resource allocation | "Forecast bed utilization next month" |
| `ai_summarize` | Clinical note summarization | "Summarize patient history for handoff" |
| `ai_translate` | Multi-language patient communication | "Translate discharge instructions to Spanish" |
| `ai_extract` | Clinical data extraction | "Extract medications from progress notes" |
| `ai_analyze_sentiment` | Patient feedback analysis | "Analyze sentiment of patient surveys" |
| `ai_similarity` | Clinical matching, research | "Find similar patient cases" |
| `ai_mask` | PHI de-identification | "Mask patient identifiers in research data" |
| `ai_fix_grammar` | Clinical documentation | "Correct grammar in clinical notes" |
| `ai_gen` | Clinical content generation | "Generate patient education materials" |

### **Multi-Function Query Processing**
```python
# Example: Complex healthcare analytics query
Query: "Classify patient risk and forecast readmissions with Spanish translations"

AI Functions Detected: ["ai_classify", "ai_forecast", "ai_translate"]
Processing: Multi-function pipeline with sequential execution
```

---

## 🔒 **Security & Performance**

### **Security Features**
- **OAuth Authentication**: Secure Databricks integration with auto-refresh
- **Input Validation**: SQL injection prevention and sanitization
- **PHI Protection**: Healthcare data privacy safeguards
- **Access Control**: Role-based access through Databricks Unity Catalog

### **Performance Optimizations**
- **Intelligent Caching**: 5-minute TTL for frequently accessed data
- **Async Processing**: Non-blocking operations for better responsiveness
- **Query Optimization**: AI-driven query enhancement
- **Connection Pooling**: Efficient database connection management

### **Enterprise Readiness**
- **Scalability**: Designed for high-volume healthcare workloads
- **Monitoring**: Comprehensive user interaction tracking
- **Audit Trail**: Complete query and response logging
- **Error Handling**: Graceful degradation and detailed error reporting

---

## 📊 **User Interaction Analytics**

The platform includes comprehensive analytics to track:

- **Query Patterns**: Most common healthcare analytics requests
- **AI Function Usage**: Which AI capabilities are most valuable
- **User Feedback**: Response quality and helpfulness metrics
- **Performance Metrics**: Query response times and success rates

### **Analytics Tables**
```sql
-- User interaction tracking
user_interactions (
    interaction_id,
    user_question,
    query_classification,
    ai_function_type,
    response_type,
    is_helpful,
    feedback_reason
)
```

---

## 🛠️ **Development Guide**

### **Project Structure**
```

├── app.py                 # Main FastAPI application
├── helper.py              # Databricks Genie API integration
├── tracking.py            # User interaction logging
├── table_extraction.py    # Unity Catalog integration
├── manual_ai_content.py   # AI function definitions
├── requirements.txt       # Python dependencies
├── databricks.yml         # Databricks bundle configuration
└── templates/
    └── index.html         # Frontend interface
```

### **Key Components**

#### **Query Processing Pipeline** (`app.py`)
- `classify_query()`: AI-powered query classification
- `classify_predictive_query()`: AI function detection
- `determine_required_columns()`: Schema-based column analysis
- `final_answer_combine()`: Predictive query processing
- `direct_genie_answer()`: Standard SQL processing

#### **Databricks Integration** (`helper.py`)
- OAuth-based authentication with auto-refresh
- Async Genie API communication
- Response caching and optimization
- SQL query extraction from responses

#### **Healthcare AI Functions** (`manual_ai_content.py`)
- Complete Databricks AI function syntax
- Healthcare-specific examples and use cases
- Function parameter documentation

### **Extending the Platform**

#### **Adding New AI Functions**
1. Add function definition to `manual_ai_content.py`
2. Update `classify_predictive_query()` classification logic
3. Test with healthcare-specific queries

#### **Custom Query Types**
1. Extend `classify_query()` with new categories
2. Implement processing logic in main query handler
3. Add response formatting for new types

---

## 🔧 **Troubleshooting**

### **Common Issues**

| Issue | Solution |
|-------|----------|
| **Authentication Errors** | Verify Databricks OAuth configuration and workspace permissions |
| **Schema Loading Failed** | Check Genie Room ID and Unity Catalog access |
| **Query Timeouts** | Increase cache TTL or verify network connectivity |
| **PDF Upload Errors** | Ensure file size under 3MB and proper PDF format |
| **AI Function Errors** | Verify manual AI content configuration |

### **Debug Features**
- Console logging for all API requests
- SQL query display for generated queries
- Detailed error messages with context
- Response type indicators for query classification

### **Environment Validation**
```bash
# Check required environment variables
python -c "
import os
required = ['SPACE_ID', 'SERVING_ENDPOINT_NAME']
missing = [var for var in required if not os.getenv(var)]
print('Missing variables:', missing if missing else 'None')
"
```

---

## 🤝 **Contributing**

We welcome contributions to enhance the platform's healthcare analytics capabilities:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/healthcare-enhancement`)
3. **Implement changes** with comprehensive testing
4. **Add documentation** for new features
5. **Submit pull request** with detailed description

### **Development Workflow**
```bash
# Setup development environment
python -m venv dev_env
source dev_env/bin/activate
pip install -r requirements.txt

# Run with development settings
python app.py

# Test the application
# Add your test cases here
```

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Databricks**: For providing the AI/ML platform and Genie capabilities
- **FastAPI**: For the high-performance web framework
- **Healthcare Community**: For inspiring innovative analytics solutions

---

## 📞 **Support**

For technical support or questions:
- 📧 **Email**: [your-email@domain.com]
- 💬 **Issues**: [GitHub Issues](../../issues)
- 📖 **Documentation**: This README and inline code comments

---

*Built with ❤️ for advancing healthcare analytics through AI innovation*
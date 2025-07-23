# Feedback Classification Application

This application uses Azure OpenAI and LangChain to automatically classify customer feedback stored in Excel files.

## Features

- Reads feedback from Excel files
- Uses Azure OpenAI for intelligent classification
- Structured output with confidence scores
- Configurable categories
- Batch processing of multiple feedback entries
- Creates sample data for testing

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Azure OpenAI:**
   - Copy `.env` file and fill in your Azure OpenAI credentials:
     - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key
     - `AZURE_OPENAI_ENDPOINT`: Your Azure OpenAI endpoint URL
     - `AZURE_OPENAI_API_VERSION`: API version (e.g., 2024-02-15-preview)
     - `AZURE_OPENAI_DEPLOYMENT_NAME`: Your deployment name

3. **Configure Categories:**
   - Edit `config.py` to define your feedback categories
   - Default categories: Positive, Negative, Neutral, Feature Request, Bug Report, Complaint, Praise, Suggestion

## Usage

### Quick Start

1. **Create sample data:**
   ```bash
   python create_sample_data.py
   ```

2. **Run classification:**
   ```bash
   python feedback_classifier.py
   ```

### Custom Excel File

Your Excel file should have a column containing feedback text (column name should include "feedback").

Example structure:
| Customer_ID | Date | Feedback | Category |
|-------------|------|----------|----------|
| CUST_1001 | 2025-01-15 | "Great product!" | |
| CUST_1002 | 2025-01-16 | "Needs improvement" | |

### Programmatic Usage

```python
from feedback_classifier import FeedbackClassifier

# Initialize classifier
classifier = FeedbackClassifier()

# Classify single feedback
result = classifier.classify_feedback("The product is amazing!")
print(f"Category: {result.category}, Confidence: {result.confidence}")

# Process Excel file
classifier.process_excel_file("input.xlsx", "output.xlsx")
```

## File Structure

```
feedback_classification/
├── requirements.txt          # Python dependencies
├── .env                     # Azure OpenAI configuration
├── config.py               # Feedback categories configuration
├── create_sample_data.py   # Generate sample Excel data
├── feedback_classifier.py  # Main classification application
└── README.md               # This file
```

## Output

The application adds two columns to your Excel file:
- **Category**: The classified category
- **Confidence**: Confidence score (0.0 to 1.0)

## Customization

### Adding New Categories

Edit `config.py` and modify the `FEEDBACK_CATEGORIES` list:

```python
FEEDBACK_CATEGORIES = [
    "Product Quality - Positive",
    "Product Quality - Negative", 
    "Customer Service - Positive",
    "Customer Service - Negative",
    "Pricing Issues",
    "Feature Request",
    "Bug Report"
]
```

### Adjusting Classification Parameters

In `feedback_classifier.py`, you can modify:
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = creative)
- Prompt template: Customize the classification instructions
- Confidence thresholds: Add validation logic

## Troubleshooting

1. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
2. **Azure OpenAI errors**: Verify your credentials in `.env` file
3. **Excel format errors**: Ensure your Excel file has a column with "feedback" in the name
4. **Memory issues**: For large files, consider processing in batches

## Dependencies

- `langchain`: LLM framework
- `langchain-openai`: Azure OpenAI integration
- `pandas`: Excel file processing
- `openpyxl`: Excel file support
- `python-dotenv`: Environment variable management
- `pydantic`: Data validation and structured output

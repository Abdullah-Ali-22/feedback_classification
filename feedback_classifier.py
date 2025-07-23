"""
Feedback Classification Application using Azure OpenAI and LangChain
"""

import os
import pandas as pd
from dotenv import load_dotenv
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from config import FEEDBACK_CATEGORIES

# Load environment variables
load_dotenv()

class FeedbackClassification(BaseModel):
    """Pydantic model for structured output from LLM"""
    category: str = Field(
        description=f"The most appropriate category for the feedback. Must be one of: {', '.join(FEEDBACK_CATEGORIES)}"
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    
    @validator('category')
    def validate_category(cls, v):
        if v not in FEEDBACK_CATEGORIES:
            raise ValueError(f'Category must be one of: {", ".join(FEEDBACK_CATEGORIES)}')
        return v

class FeedbackClassifier:
    """Main class for feedback classification using Azure OpenAI"""
    
    def __init__(self):
        """Initialize the Azure OpenAI client and setup the classification chain"""
        self.llm = self._setup_azure_openai()
        self.parser = PydanticOutputParser(pydantic_object=FeedbackClassification)
        self.prompt = self._create_prompt()
        self.chain = self.prompt | self.llm | self.parser
    
    def _setup_azure_openai(self) -> AzureChatOpenAI:
        """Setup Azure OpenAI client with environment variables"""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not all([api_key, endpoint, api_version, deployment_name]):
            raise ValueError("Missing required Azure OpenAI environment variables. Please check your .env file.")
        
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment=deployment_name,
            temperature=0.1  # Low temperature for consistent classification
        )
    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the prompt template for feedback classification"""
        categories_str = ", ".join(FEEDBACK_CATEGORIES)
        
        prompt_template = """
        You are an expert feedback classifier. Your task is to classify customer feedback into one of the predefined categories.
        
        Available Categories: {categories}
        
        Customer Feedback: {feedback}
        
        Instructions:
        1. Analyze the sentiment and content of the feedback
        2. Choose the MOST APPROPRIATE category from the available categories
        3. Provide a confidence score between 0.0 and 1.0
        4. Be consistent in your classifications
        
        {format_instructions}
        """
        
        return ChatPromptTemplate.from_template(prompt_template).partial(
            categories=categories_str,
            format_instructions=self.parser.get_format_instructions()
        )
    
    def classify_feedback(self, feedback: str) -> FeedbackClassification:
        """Classify a single feedback text"""
        try:
            result = self.chain.invoke({"feedback": feedback})
            
            # The validator ensures the category is always valid
            # No additional validation needed since Pydantic handles it
            return result
        except Exception as e:
            print(f"Error classifying feedback: {e}")
            # Return "Other" as fallback category (or first category if "Other" not in list)
            fallback_category = "Other" if "Other" in FEEDBACK_CATEGORIES else FEEDBACK_CATEGORIES[0]
            return FeedbackClassification(category=fallback_category, confidence=0.0)
    
    def process_excel_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Process an Excel file containing feedback and add classifications
        
        Args:
            input_file: Path to the input Excel file
            output_file: Path to the output Excel file (if None, overwrites input file)
        
        Returns:
            Path to the output file
        """
        if output_file is None:
            output_file = input_file
        
        # Read the Excel file
        try:
            df = pd.read_excel(input_file)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")
        
        # Check if feedback column exists
        feedback_columns = [col for col in df.columns if 'feedback' in col.lower()]
        if not feedback_columns:
            raise ValueError("No feedback column found. Please ensure your Excel file has a column containing 'feedback' in its name.")
        
        feedback_column = feedback_columns[0]
        print(f"Using column '{feedback_column}' for feedback text")
        
        # Initialize results columns if they don't exist
        if 'Category' not in df.columns:
            df['Category'] = ''
        if 'Confidence' not in df.columns:
            df['Confidence'] = 0.0
        
        # Process each feedback
        total_rows = len(df)
        print(f"Processing {total_rows} feedback entries...")
        
        for index, row in df.iterrows():
            feedback_text = row[feedback_column]
            
            if pd.isna(feedback_text) or feedback_text.strip() == '':
                continue
            
            print(f"Processing row {index + 1}/{total_rows}: {feedback_text[:50]}...")
            
            try:
                classification = self.classify_feedback(feedback_text)
                df.at[index, 'Category'] = classification.category
                df.at[index, 'Confidence'] = classification.confidence
            except Exception as e:
                print(f"Error processing row {index + 1}: {e}")
                df.at[index, 'Category'] = 'Error'
                df.at[index, 'Confidence'] = 0.0
        
        # Save the results
        df.to_excel(output_file, index=False)
        print(f"Classification complete! Results saved to: {output_file}")
        
        # Print summary
        category_counts = df['Category'].value_counts()
        print("\nClassification Summary:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        
        return output_file

def main():
    """Main function to run the feedback classification"""
    classifier = FeedbackClassifier()
    
    # Example usage change here the real path of the excel file
    input_file = "sample_feedback.xlsx"
    
    # Check if sample file exists, create if not
    if not os.path.exists(input_file):
        print("Sample feedback file not found. Creating one...")
        from create_sample_data import create_sample_feedback_file
        create_sample_feedback_file(input_file)
    
    # Process the file
    try:
        output_file = classifier.process_excel_file(input_file, "classified_feedback.xlsx")
        print(f"\nFeedback classification completed successfully!")
        print(f"Check the output file: {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

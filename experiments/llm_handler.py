"""
LLM Handler for non-banking queries.
Wraps Groq LLM and formats responses for banking chatbot context.
"""

from typing import Optional
from dotenv import load_dotenv
import os

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Load environment variables
load_dotenv()


class LLMHandler:
    """Handle non-banking queries using Groq LLM."""
    
    def __init__(self):
        """Initialize Groq LLM client."""
        self.api_key = os.getenv("GROQ_API_KEY")
        self.llm = None
        self.available = False
        
        if LANGCHAIN_AVAILABLE and self.api_key:
            try:
                self.llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    temperature=0.3,
                    api_key=self.api_key
                )
                self.available = True
            except Exception as e:
                print(f"⚠️ Failed to initialize Groq LLM: {e}")
                self.available = False
    
    def generate(self, user_text: str) -> str:
        """
        Generate response for non-banking query using Groq LLM.
        
        Args:
            user_text: User input text
            
        Returns:
            Formatted LLM response or fallback message
        """
        if not self.available:
            return self._fallback_response()
        
        try:
            # Add banking chatbot context to prompt
            prompt = f"""You are BankBot AI, a banking assistant chatbot. 
The user is asking a non-banking question. Answer concisely and helpfully.
If relevant, suggest they ask you about banking services.

User question: {user_text}"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            
            # Format with LLM indicator
            return f"🌐 **General Information (LLM Generated)**\n\n{content}"
        
        except Exception as e:
            print(f"❌ Groq LLM error: {e}")
            return self._fallback_response()
    
    def _fallback_response(self) -> str:
        """Fallback response when LLM is unavailable."""
        return (
            "🌐 **General Information**\n\n"
            "That's an interesting question! I'm specialized in banking operations. "
            "Feel free to ask me about:\n"
            "- Checking your balance\n"
            "- Transferring money\n"
            "- Blocking/unblocking cards\n"
            "- Finding ATMs\n"
            "- Loan information"
        )


# Singleton instance
_llm_handler = None


def get_llm_handler() -> LLMHandler:
    """Get or create singleton LLM handler instance."""
    global _llm_handler
    if _llm_handler is None:
        _llm_handler = LLMHandler()
    return _llm_handler


def generate_llm_response(user_text: str) -> str:
    """
    Convenience function to generate LLM response.
    
    Args:
        user_text: User input text
        
    Returns:
        Formatted LLM response
    """
    handler = get_llm_handler()
    return handler.generate(user_text)

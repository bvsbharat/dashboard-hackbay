"""
DeepAgent for intelligent document analysis using LangChain and LangGraph
Inspired by https://github.com/langchain-ai/deepagents
"""

import os
import logging
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document

logger = logging.getLogger(__name__)


class FinancialDeepAgent:
    """Deep agent for financial document analysis"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.llm = None

        if self.api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                api_key=self.api_key
            )
            logger.info("Financial DeepAgent initialized")

    async def analyze_document(
        self,
        document_text: str,
        document_type: str = "financial"
    ) -> Dict[str, Any]:
        """
        Analyze a financial document and extract structured information

        Args:
            document_text: Full text of the document
            document_type: Type of document (financial, legal, etc.)

        Returns:
            Structured analysis with metrics, insights, and red flags
        """
        if not self.llm:
            return self._get_empty_analysis()

        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(document_text, document_type)

            # Get LLM response
            response = await self.llm.ainvoke(prompt)

            # Parse and structure the response
            analysis = self._parse_analysis_response(response.content)

            logger.info("Document analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return self._get_empty_analysis()

    def _create_analysis_prompt(
        self,
        document_text: str,
        document_type: str
    ) -> List:
        """Create a structured prompt for document analysis"""

        system_message = """You are a financial analyst AI specialized in due diligence.
Your task is to analyze financial documents and extract:
1. Key financial metrics (revenue, expenses, profit, assets, liabilities, equity)
2. Important insights and trends
3. Potential red flags or concerns
4. Risk assessment

Return your analysis in a structured JSON format."""

        user_message = f"""Analyze this {document_type} document and provide a comprehensive analysis.

Document Text:
{document_text[:10000]}  # Limit to first 10k chars for API limits

IMPORTANT INSTRUCTIONS FOR NUMBER EXTRACTION:
- Extract ALL numbers EXACTLY as they appear in the document
- For "millions" ($X million or $X.XM), multiply by 1,000,000
- For "billions" ($X billion or $X.XB), multiply by 1,000,000,000
- Example: "$44.1 billion" → 44100000000 (as raw number)
- Example: "$44,062 million" → 44062000000 (remove commas, multiply by 1M)
- Example: "Revenue of $26,044" (in millions) → 26044000000
- Return NULL only if the metric truly doesn't exist in the document

Please extract and return in JSON format:
{{
    "metrics": {{
        "revenue": <number in dollars or null>,
        "expenses": <number in dollars or null>,
        "profit": <number in dollars or null>,
        "assets": <number in dollars or null>,
        "liabilities": <number in dollars or null>,
        "equity": <number in dollars or null>
    }},
    "insights": [
        "insight 1",
        "insight 2",
        ...
    ],
    "red_flags": [
        {{
            "severity": "high|medium|low",
            "category": "Financial|Legal|Operational",
            "title": "Issue title",
            "description": "Detailed description",
            "evidence": "Supporting evidence from document"
        }}
    ],
    "risk_score": <number 0-100>,
    "summary": "Brief summary of the document"
}}
"""

        return [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]

    def _parse_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response into structured format with validation"""
        import json
        import re

        logger.info("🔍 Parsing DeepAgent response...")
        logger.debug(f"Raw response (first 500 chars): {response_text[:500]}")

        try:
            # Try to extract JSON from response (look for code blocks too)
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # JSON code block
                r'```\s*(.*?)\s*```',       # Generic code block
                r'\{.*\}',                  # Raw JSON object
            ]

            analysis = None
            for pattern in json_patterns:
                json_match = re.search(pattern, response_text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(1) if json_match.lastindex else json_match.group()
                        analysis = json.loads(json_str)
                        logger.info("✅ Successfully extracted JSON from response")
                        break
                    except json.JSONDecodeError:
                        continue

            if not analysis:
                logger.warning("❌ No valid JSON found in response")
                return self._fallback_parse(response_text)

            # Validate and clean metrics
            if "metrics" in analysis:
                metrics = analysis["metrics"]
                logger.info(f"📊 Raw metrics extracted: {metrics}")

                # Ensure all metric values are numbers or None
                cleaned_metrics = {}
                for key, value in metrics.items():
                    if value is None or value == "null":
                        cleaned_metrics[key] = None
                    elif isinstance(value, (int, float)):
                        cleaned_metrics[key] = value
                    elif isinstance(value, str):
                        # Try to parse string numbers
                        try:
                            cleaned_metrics[key] = float(value.replace(",", "").replace("$", ""))
                        except (ValueError, AttributeError):
                            cleaned_metrics[key] = None
                    else:
                        cleaned_metrics[key] = None

                analysis["metrics"] = cleaned_metrics
                logger.info(f"✅ Cleaned metrics: {cleaned_metrics}")

            return analysis

        except Exception as e:
            logger.error(f"❌ Failed to parse analysis response: {e}")
            logger.debug(f"Response text: {response_text}")
            return self._fallback_parse(response_text)

    def _fallback_parse(self, response_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        return {
            "metrics": {
                "revenue": None,
                "expenses": None,
                "profit": None,
                "assets": None,
                "liabilities": None,
                "equity": None
            },
            "insights": [
                "Analysis completed but structured data extraction failed",
                "Raw analysis available in summary"
            ],
            "red_flags": [],
            "risk_score": 50,
            "summary": response_text[:500]
        }

    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            "metrics": {
                "revenue": None,
                "expenses": None,
                "profit": None,
                "assets": None,
                "liabilities": None,
                "equity": None
            },
            "insights": [],
            "red_flags": [],
            "green_flags": [],
            "risk_score": 0,
            "summary": "Analysis not available"
        }

    async def analyze_with_context(
        self,
        document_text: str,
        context_documents: List[Dict[str, Any]] = None,
        document_type: str = "financial"
    ) -> Dict[str, Any]:
        """
        Analyze a document with contextual information from similar documents

        Args:
            document_text: Text of the new document
            context_documents: List of similar documents from RAG {text, metadata, score}
            document_type: Type of document

        Returns:
            Enhanced analysis with red flags, green flags, and comparative insights
        """
        if not self.llm:
            return self._get_empty_analysis()

        try:
            # Create enhanced prompt with context
            prompt = self._create_contextual_analysis_prompt(
                document_text,
                context_documents or [],
                document_type
            )

            # Get LLM response
            response = await self.llm.ainvoke(prompt)

            # Parse and structure the response
            analysis = self._parse_analysis_response(response.content)

            # Ensure green_flags field exists
            if "green_flags" not in analysis:
                analysis["green_flags"] = []

            logger.info("Context-aware document analysis completed")
            return analysis

        except Exception as e:
            logger.error(f"Context-aware analysis failed: {e}")
            return self._get_empty_analysis()

    def _create_contextual_analysis_prompt(
        self,
        document_text: str,
        context_documents: List[Dict[str, Any]],
        document_type: str
    ) -> List:
        """Create enhanced prompt with historical context"""

        system_message = """You are a financial analyst AI specialized in due diligence.
Your task is to analyze financial documents WITH CONTEXT from previously analyzed documents.

You must extract:
1. Key financial metrics (revenue, expenses, profit, assets, liabilities, equity)
2. Important insights and trends
3. RED FLAGS - Potential concerns, risks, and warning signs
4. GREEN FLAGS - Positive indicators, strengths, and opportunities
5. Comparative analysis against the provided context documents
6. Risk assessment

Return your analysis in a structured JSON format."""

        # Build context section
        context_section = ""
        if context_documents:
            context_section = "\n\n### HISTORICAL CONTEXT FROM SIMILAR DOCUMENTS:\n"
            for i, doc in enumerate(context_documents[:3], 1):
                doc_text = doc.get("text", "")[:1000]  # Limit context length
                metadata = doc.get("metadata", {})
                filename = metadata.get("filename", f"Document {i}")
                context_section += f"\n**Reference Document {i}: {filename}**\n{doc_text}\n"

        user_message = f"""Analyze this NEW {document_type} document and provide a comprehensive analysis.

{context_section}

### NEW DOCUMENT TO ANALYZE:
{document_text[:10000]}

CRITICAL INSTRUCTIONS FOR NUMBER EXTRACTION:
- Extract ALL numbers EXACTLY as they appear in the document
- For "millions" ($X million or $X.XM), multiply by 1,000,000
- For "billions" ($X billion or $X.XB), multiply by 1,000,000,000
- Example: "$44.1 billion" → 44100000000 (as raw number)
- Example: "$44,062 million" → 44062000000 (remove commas, multiply by 1M)
- Example: "Revenue of $26,044" (in millions table) → 26044000000
- Return NULL only if the metric truly doesn't exist in the document

Please extract and return in JSON format:
{{
    "company_info": {{
        "name": "<company name or null>",
        "industry": "<industry/sector or null>",
        "fiscal_period": "<FY2024 Q1, etc or null>",
        "employees": <number or null>,
        "founded": "<year or null>"
    }},
    "metrics": {{
        "revenue": <number in dollars or null>,
        "expenses": <number in dollars or null>,
        "profit": <number in dollars or null>,
        "gross_profit": <number in dollars or null>,
        "operating_profit": <number in dollars or null>,
        "ebitda": <number in dollars or null>,
        "assets": <number in dollars or null>,
        "current_assets": <number in dollars or null>,
        "liabilities": <number in dollars or null>,
        "current_liabilities": <number in dollars or null>,
        "equity": <number in dollars or null>,
        "cash": <number in dollars or null>,
        "debt": <number in dollars or null>,
        "operating_cash_flow": <number in dollars or null>,
        "free_cash_flow": <number in dollars or null>,
        "revenue_growth_yoy": <percentage or null>,
        "customer_count": <number or null>
    }},
    "calculated_ratios": {{
        "gross_margin": <percentage or null>,
        "operating_margin": <percentage or null>,
        "net_margin": <percentage or null>,
        "current_ratio": <number or null>,
        "quick_ratio": <number or null>,
        "debt_to_equity": <number or null>,
        "roe": <percentage or null>,
        "roa": <percentage or null>
    }},
    "insights": [
        "Key insight 1 (include comparisons to context documents when relevant)",
        "Key insight 2",
        ...
    ],
    "red_flags": [
        {{
            "severity": "high|medium|low",
            "category": "Financial|Legal|Operational|Compliance",
            "title": "Issue title",
            "description": "Detailed description",
            "evidence": "Supporting evidence from document",
            "recommendation": "Suggested action or mitigation"
        }}
    ],
    "green_flags": [
        {{
            "category": "Financial Strength|Growth|Efficiency|Market Position",
            "title": "Positive indicator title",
            "description": "Detailed description",
            "evidence": "Supporting evidence from document",
            "impact": "Why this is beneficial"
        }}
    ],
    "risk_score": <number 0-100>,
    "summary": "Brief summary including how this document compares to context documents"
}}

IMPORTANT:
- Be thorough in identifying BOTH red flags AND green flags
- Use context documents for comparison when available
- Green flags should highlight strengths, competitive advantages, growth indicators
- Red flags should identify risks, concerns, and areas requiring attention
"""

        return [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]

    async def answer_question(
        self,
        question: str,
        context_documents: List[str],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question about documents using DeepAgent approach

        Args:
            question: User's question
            context_documents: List of relevant document texts
            chat_history: Optional chat history for context

        Returns:
            Answer with sources and confidence
        """
        if not self.llm:
            return {
                "answer": "DeepAgent not initialized",
                "sources": [],
                "confidence": 0.0
            }

        try:
            # Prepare context
            context = "\n\n".join([
                f"Document {i+1}:\n{doc[:2000]}"
                for i, doc in enumerate(context_documents[:5])
            ])

            # Create prompt with chain-of-thought reasoning
            system_prompt = """You are a financial due diligence assistant with deep analytical capabilities.

When answering questions:
1. Carefully analyze the provided documents
2. Think step-by-step about the question
3. Provide accurate answers with specific evidence
4. Cite sources with document references
5. If uncertain, acknowledge limitations

Always provide factual, evidence-based responses."""

            user_prompt = f"""Context Documents:
{context}

Question: {question}

Please provide a detailed answer with:
1. Direct answer to the question
2. Supporting evidence from the documents
3. Any relevant caveats or additional context

Answer:"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            # Get response
            response = await self.llm.ainvoke(messages)

            return {
                "answer": response.content,
                "sources": [
                    {
                        "document": f"Document {i+1}",
                        "relevance": "high" if i < 2 else "medium"
                    }
                    for i in range(min(len(context_documents), 3))
                ],
                "confidence": 0.85 if context_documents else 0.5
            }

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }


# Singleton instance
_deep_agent = None

def get_deep_agent() -> FinancialDeepAgent:
    """Get or create DeepAgent instance"""
    global _deep_agent
    if _deep_agent is None:
        _deep_agent = FinancialDeepAgent()
    return _deep_agent

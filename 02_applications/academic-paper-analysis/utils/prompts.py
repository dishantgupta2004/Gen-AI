"""
Prompt templates for different types of academic content summarization.
"""

from langchain.prompts import PromptTemplate
from typing import Dict

class AcademicPrompts:
    """Collection of prompt templates for academic content analysis."""
    
    @staticmethod
    def get_comprehensive_prompt() -> PromptTemplate:
        """Comprehensive academic paper analysis prompt."""
        template = """
You are an expert academic researcher tasked with analyzing the following research content. 
Provide a comprehensive analysis structured as follows:

## Abstract Summary
Provide a concise 150-word summary of the main research question, approach, and findings.

## Key Findings
List 5-7 most important discoveries, results, or conclusions from this research.
- Use bullet points
- Focus on novel contributions
- Include quantitative results where available

## Methodology
Summarize the research methods, experimental design, and analytical approaches used.
- Research design type
- Data collection methods  
- Analysis techniques
- Sample size/scope if applicable

## Future Research Directions
Identify 3-5 areas for future investigation suggested by this work.
- Unresolved questions
- Potential extensions
- Methodological improvements
- Broader applications

Content to analyze:
{text}

Please ensure your analysis is scholarly, precise, and focuses on the academic contributions.
"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    @staticmethod
    def get_executive_prompt() -> PromptTemplate:
        """Executive summary focused prompt."""
        template = """
You are tasked with creating an executive summary of academic research for busy professionals and decision-makers.

Provide a clear, accessible analysis in two sections:

## Executive Summary (200 words)
Write a non-technical summary that explains:
- What problem does this research address?
- What did the researchers do?
- What are the main findings?
- Why do these findings matter?

## Key Takeaways
Provide 5-6 actionable insights in bullet points:
- Focus on practical implications
- Include measurable outcomes where available
- Highlight potential applications
- Note any limitations or caveats

Content to analyze:
{text}

Keep the language accessible while maintaining academic rigor. Avoid jargon and explain technical terms.
"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    @staticmethod
    def get_research_focus_prompt() -> PromptTemplate:
        """Research methodology and findings focused prompt."""
        template = """
You are a research methodologist analyzing academic work. Focus on the technical and methodological aspects.

Provide detailed analysis in these sections:

## Research Methodology
Analyze the research approach:
- Study design and rationale
- Data collection procedures
- Analytical methods and tools
- Quality control measures
- Limitations and assumptions

## Key Findings & Results
Present the main results:
- Primary outcomes (with statistics if available)
- Secondary findings
- Unexpected discoveries
- Statistical significance and effect sizes
- Visual data interpretations

## Future Research Opportunities
Identify research gaps and extensions:
- Methodological improvements needed
- Unexplored variables or populations
- Replication opportunities
- Interdisciplinary connections
- Technology or tool developments needed

Content to analyze:
{text}

Focus on technical accuracy and methodological rigor in your analysis.
"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    @staticmethod
    def get_map_reduce_prompt() -> PromptTemplate:
        """Prompt for map-reduce summarization strategy."""
        template = """
Summarize the following section of an academic paper. Focus on:
- Main arguments and findings in this section
- Key data or evidence presented
- Methodological details if present
- Important conclusions or implications

Keep your summary concise but comprehensive (200-300 words).

Content:
{text}

Summary:
"""
        return PromptTemplate(template=template, input_variables=["text"])
    
    @staticmethod
    def get_refine_prompt() -> PromptTemplate:
        """Prompt for refine summarization strategy."""
        template = """
You have been provided with an existing summary and additional context from an academic paper.
Your task is to refine and improve the existing summary by incorporating new information.

Existing Summary:
{existing_answer}

Additional Context:
{text}

Please provide an improved, more comprehensive summary that:
1. Integrates the new information seamlessly
2. Maintains academic rigor and precision
3. Ensures no important details are lost
4. Keeps the summary well-structured and coherent

Refined Summary:
"""
        return PromptTemplate(
            template=template, 
            input_variables=["existing_answer", "text"]
        )
    
    @staticmethod
    def get_final_combination_prompt() -> PromptTemplate:
        """Prompt for combining multiple summaries into final output."""
        template = """
You have multiple summaries of different sections from the same academic work. 
Combine them into a coherent, comprehensive academic digest.

Summaries to combine:
{text}

Create a unified analysis that:
1. Eliminates redundancy
2. Maintains logical flow
3. Preserves all key insights
4. Follows academic writing standards
5. Structures information clearly with appropriate headers

Provide the final comprehensive academic digest:
"""
        return PromptTemplate(template=template, input_variables=["text"])

def get_prompt_by_type(summary_type: str) -> PromptTemplate:
    """Get the appropriate prompt template based on summary type."""
    prompt_mapping = {
        "Comprehensive": AcademicPrompts.get_comprehensive_prompt(),
        "Executive": AcademicPrompts.get_executive_prompt(),
        "Research Focus": AcademicPrompts.get_research_focus_prompt()
    }
    
    return prompt_mapping.get(summary_type, AcademicPrompts.get_comprehensive_prompt())
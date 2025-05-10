"""Code generation for autogenlib using OpenAI API."""

import openai
import os
import ast
from ._cache import get_all_modules, get_cached_prompt
from logging import getLogger

logger = getLogger(__name__)


def validate_code(code):
    """Validate the generated code against PEP standards."""
    try:
        # Check if the code is syntactically valid
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_codebase_context():
    """Get the full codebase context for all cached modules."""
    modules = get_all_modules()

    if not modules:
        return ""

    context = "Here is the existing codebase for reference:\n\n"

    for module_name, data in modules.items():
        if "code" in data:
            context += f"# Module: {module_name}\n```python\n{data['code']}\n```\n\n"

    return context


def generate_code(description, fullname, existing_code=None, caller_info=None):
    """Generate code using the OpenAI API."""
    parts = fullname.split(".")
    if len(parts) < 2:
        return None

    module_name = parts[1]
    function_name = parts[2] if len(parts) > 2 else None

    # Get the cached prompt or use the provided description
    module_to_check = ".".join(fullname.split(".")[:2])  # e.g., 'autogenlib.totp'
    cached_prompt = get_cached_prompt(module_to_check)
    current_description = cached_prompt or description

    # Get the full codebase context
    codebase_context = get_codebase_context()

    # Add caller code context if available
    caller_context = ""
    if caller_info and caller_info.get("code"):
        code = caller_info.get("code", "")
        # Extract the most relevant parts of the code if possible
        # Try to focus on the sections that use the requested module/function
        relevant_parts = []
        module_parts = fullname.split(".")

        if len(module_parts) >= 2:
            # Look for imports of this module
            module_prefix = f"from {module_parts[0]}.{module_parts[1]}"
            import_lines = [line for line in code.split("\n") if module_prefix in line]
            if import_lines:
                relevant_parts.extend(import_lines)

            # Look for usages of the imported functions
            if len(module_parts) >= 3:
                func_name = module_parts[2]
                func_usage_lines = [
                    line
                    for line in code.split("\n")
                    if func_name in line and not line.startswith(("import ", "from "))
                ]
                if func_usage_lines:
                    relevant_parts.extend(func_usage_lines)

        # Include relevant parts if found, otherwise use the whole code
        if relevant_parts:
            caller_context = f"""
            Here is the code that is importing and using this module/function:
            ```python
            # File: {caller_info.get("filename", "unknown")}
            # --- Relevant snippets ---
            {"\n".join(relevant_parts)}
            ```
            
            And here is the full context:
            ```python
            {code}
            ```
            
            Pay special attention to how the requested functionality will be used in the code snippets above.
            """
        else:
            caller_context = f"""
            Here is the code that is importing this module/function:
            ```python
            # File: {caller_info.get("filename", "unknown")}
            {code}
            ```
            
            Pay special attention to how the requested functionality will be used in this code.
            """

        logger.debug(f"Including caller context from {caller_info.get('filename')}")

    # Create a prompt for the OpenAI API
    if function_name and existing_code:
        prompt = f"""
        You are extending an existing Python module named '{module_name}'.
        
        The overall purpose of this library is:
        {current_description}
        
        Here is the existing code for this module:
        ```python
        {existing_code}
        ```
        
        {codebase_context}
        
        {caller_context}
        
        Add a new function/class named '{function_name}' (if it is capitalized - you should generate class) that implements the following functionality:
        {description}
        
        Keep all existing functions and classes intact. Follow PEP 8 style guidelines.
        Provide the complete module code including both existing functionality and the new function.
        Return ONLY the Python code for this module without any explanations or markdown.
        """
    elif function_name:
        prompt = f"""
        Create a Python module named '{module_name}' with a function/class named '{function_name}' (if it is capitalized - you should generate class) that implements the following functionality:
        {description}
        
        {codebase_context}
        
        {caller_context}
        
        Follow PEP 8 style guidelines. Provide only the Python code without any explanations.
        """
    else:
        prompt = f"""
        Create a Python package named '{module_name}' that implements the following functionality:
        {description}
        
        {codebase_context}
        
        {caller_context}
        
        Follow PEP 8 style guidelines. Provide only the Python code without any explanations.
        Do not generate any additional Python files and do not add file path to your answer.
        """

    try:
        # Set API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable.")

        base_url = os.environ.get("OPENAI_API_BASE_URL")
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        logger.debug("Prompt: %s", prompt)

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that generates Python code for requested modules. "
                        "You have two main tasks: (1) understand the calling context and (2) generate compatible code.\n\n"
                        "1. CONTEXT ANALYSIS:\n"
                        "- Carefully analyze caller code to understand data structures and usage patterns\n"
                        "- Identify variable types and function signatures from usage examples\n"
                        "- Ensure generated functions work with the exact data structure shown in the caller code\n"
                        "- Pay attention to naming conventions in the caller code\n\n"
                        "2. CODE GENERATION:\n"
                        "- Ensure all code strictly follows PEP standards and established best practices\n"
                        "- Use ONLY the Python standard library, and do not import any third-party libraries\n"
                        "- ONLY import modules that have already been defined within this library\n"
                        "- Generate code with appropriate error handling and documentation\n"
                        "- Make reasonable assumptions when information is missing\n\n"
                        "HANDLING NESTED IMPORTS:\n"
                        "- If your code needs to use functionality from another module in this library, properly import it\n"
                        "- If you need to import a module that doesn't exist yet, import it as if it did exist\n"
                        "- The import system will handle generating any imported modules that don't exist yet\n\n"
                        "RESPONSE FORMAT:\n"
                        "Respond ONLY with the complete Python code for the requested module—no explanations or text outside the code."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        # Get the generated code
        code = response.choices[0].message.content.strip()

        logger.debug("Answer: %s", code)

        # Remove markdown code blocks if present
        if code.startswith("```python"):
            code = code.replace("```python", "", 1)
            code = code.replace("```", "", 1)
        elif code.startswith("```"):
            code = code.replace("```", "", 1)
            code = code.replace("```", "", 1)

        code = code.strip()

        # Validate the code
        if validate_code(code):
            return code
        else:
            print("Generated code is not valid.")
            return None
    except Exception as e:
        print(f"Error generating code: {e}")
        return None

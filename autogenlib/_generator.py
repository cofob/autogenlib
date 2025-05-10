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


def check_if_empty_module_needed(fullname, caller_info):
    """Check if the requested module appears to be just for namespacing.

    Args:
        fullname: The full name of the module being imported
        caller_info: Information about the calling code

    Returns:
        bool: True if the module appears to be only for namespacing
    """
    if not caller_info or not caller_info.get("code"):
        return False

    # Check if this looks like a nested import
    parts = fullname.split(".")
    if len(parts) <= 2:  # Not a nested module
        return False

    code = caller_info.get("code", "")

    # If the import is like: from autogenlib.x.y import z
    # and there's no direct usage of x or x.y, it might be just namespacing
    module_name = ".".join(parts[:2])  # e.g., 'autogenlib.crypto'

    # Check for direct usage of the module (not just imports)
    if f"{parts[1]}." in code and not all(
        f"{parts[1]}." in line
        for line in code.split("\n")
        if f"{parts[1]}." in line and "import" in line
    ):
        return False  # There's direct usage, so it's not just namespacing

    # Check for import patterns that suggest nested modules
    parent_module = f"from {parts[0]}.{parts[1]}"
    child_modules = [
        line for line in code.split("\n") if parent_module in line and "import" in line
    ]

    # If there are nested imports and no direct usage, it's likely just namespacing
    return len(child_modules) > 0


def generate_code(description, fullname, existing_code=None, caller_info=None):
    """Generate code using the OpenAI API."""
    parts = fullname.split(".")
    if len(parts) < 2:
        return None

    module_name = parts[1]
    function_name = parts[2] if len(parts) > 2 else None

    # Check if this might be an empty module (just for namespacing)
    is_empty_module = check_if_empty_module_needed(fullname, caller_info)

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
            
            Analyze how the requested functionality will be used in the code snippets above.
            Pay special attention to parameter types, return values, and expected behavior.
            """
        else:
            caller_context = f"""
            Here is the code that is importing this module/function:
            ```python
            # File: {caller_info.get("filename", "unknown")}
            {code}
            ```
            
            Analyze how the requested functionality will be used in this code.
            Pay special attention to parameter types, return values, and expected behavior.
            """

        logger.debug(f"Including caller context from {caller_info.get('filename')}")

    # Add empty module detection info
    empty_module_context = ""
    if is_empty_module:
        empty_module_context = f"""
        IMPORTANT: Based on the calling code analysis, this module '{fullname}' appears to be 
        primarily used for namespacing/structural purposes rather than for its own functionality.
        
        If appropriate, you may generate a minimal module with just necessary imports and docstrings,
        but without substantial implementation if the code doesn't show direct usage of this module.
        """

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
        
        {empty_module_context}
        
        I need you to add a new {"function" if not function_name[0].isupper() else "class"} named '{function_name}' that implements the following functionality:
        {description}
        
        IMPORTANT REQUIREMENTS:
        1. Maintain consistency with the existing code style, naming patterns, and error handling
        2. Keep all existing functions and classes intact
        3. Add comprehensive docstrings with parameters, returns, and exceptions
        4. Use type hints for better code clarity when appropriate
        5. Analyze the caller code carefully to ensure the function works with existing data structures
        6. If analysis suggests this is purely for structural imports with no actual functionality needed,
           provide minimal implementation with appropriate comments
        
        Return the COMPLETE module code including both existing functionality and the new function.
        Your response must be ONLY valid Python code without any explanations or markdown.
        """
    elif function_name:
        prompt = f"""
        Create a Python module named '{module_name}' with a {"function" if not function_name[0].isupper() else "class"} named '{function_name}' that implements the following functionality:
        {description}
        
        {codebase_context}
        
        {caller_context}
        
        {empty_module_context}
        
        IMPORTANT REQUIREMENTS:
        1. Start with a clear module docstring explaining the purpose
        2. Include detailed docstrings with parameters, returns, and exceptions
        3. Use type hints when appropriate for clarity
        4. Implement robust error handling with specific exception types
        5. Follow strict PEP 8 style guidelines
        6. Analyze the caller context carefully to match the expected usage pattern
        7. If analysis suggests this is purely for structural imports with no actual functionality needed,
           provide minimal implementation with appropriate comments
        
        Your response must be ONLY valid Python code without any explanations or markdown.
        """
    else:
        prompt = f"""
        Create a Python package named '{module_name}' that implements the following functionality:
        {description}
        
        {codebase_context}
        
        {caller_context}
        
        {empty_module_context}
        
        IMPORTANT REQUIREMENTS:
        1. Start with a clear module docstring explaining the purpose
        2. Implement functions/classes that fulfill the described purpose
        3. Include detailed docstrings with parameters, returns, and exceptions
        4. Use type hints when appropriate for clarity
        5. Follow strict PEP 8 style guidelines
        6. Analyze the caller context to structure the package in a way that matches expected usage
        7. If analysis suggests this is purely for structural imports with no actual functionality needed,
           provide minimal implementation with appropriate comments
        
        Do not generate any additional Python files. Provide only the Python code without explanations.
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

        # logger.debug("Prompt: %s", prompt)

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Python code generator specializing in creating modules on-demand. "
                        "Your mission is to analyze context thoroughly and produce high-quality Python code "
                        "that seamlessly integrates with existing codebase.\n\n"
                        "1. CONTEXT ANALYSIS (CRUCIAL):\n"
                        "- Deeply analyze caller code to understand exact data structures, types, and usage patterns\n"
                        "- Pay special attention to how the imported function will be used - parameter types, return values\n"
                        "- Identify naming conventions, coding style, and error handling approaches in existing code\n"
                        "- Determine if the import is simply a structural one (nested module) that doesn't require actual code\n\n"
                        "2. CODE GENERATION PRINCIPLES:\n"
                        "- Follow strict PEP 8 standards with consistent formatting\n"
                        "- Use only Python standard library (no third-party packages)\n"
                        "- Only import modules already defined within this library\n"
                        "- Create efficient implementations that handle edge cases\n\n"
                        "3. EMPTY MODULE DETECTION:\n"
                        "- If the import appears to be a nested/structural import and analysis of caller code suggests "
                        "no actual functionality is needed, return an empty module with just appropriate comments\n"
                        "- For example, if code shows `from autogenlib.crypto.hash import md5` but doesn't show usage "
                        "of autogenlib.crypto directly, the crypto module might just be organizational\n\n"
                        "RESPONSE FORMAT:\n"
                        "Provide ONLY valid Python code—no explanations, markdown, or surrounding text. "
                        "The code should be ready to execute immediately."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        # Get the generated code
        code = response.choices[0].message.content.strip()

        # logger.debug("Answer: %s", code)

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

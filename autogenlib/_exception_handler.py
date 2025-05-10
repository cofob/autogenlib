"""Exception handling and LLM fix suggestions for autogenlib."""

import sys
import traceback
import os
from logging import getLogger
import openai
import time
import textwrap

from ._cache import get_cached_code, cache_module
from ._context import set_module_context
from ._state import description, exception_handler_enabled

logger = getLogger(__name__)


def setup_exception_handler():
    """Set up the global exception handler."""
    # Store the original excepthook
    original_excepthook = sys.excepthook

    # Define our custom exception hook
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        if exception_handler_enabled:
            handle_exception(exc_type, exc_value, exc_traceback)
        # Call the original excepthook regardless
        original_excepthook(exc_type, exc_value, exc_traceback)

    # Set our custom excepthook as the global handler
    sys.excepthook = custom_excepthook


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle an exception by sending it to the LLM for fix suggestions."""
    # Extract the traceback information
    tb_frames = traceback.extract_tb(exc_traceback)
    tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

    # Determine the source of the exception
    is_autogenlib_exception = False
    module_name = None
    source_code = None
    source_file = None

    # Try to find the frame where the exception originated
    for frame in tb_frames:
        filename = frame.filename
        lineno = frame.lineno

        # Check if this file is from an autogenlib module
        if "<string>" not in filename and filename != "<stdin>":
            # This is a real file
            if filename.endswith(".py"):
                source_file = filename
                module_name_from_frame = None

                # Try to get the module name from the frame
                frame_module = None
                if hasattr(frame, "frame") and hasattr(frame.frame, "f_globals"):
                    module_name_from_frame = frame.frame.f_globals.get("__name__")
                elif len(frame) > 3 and hasattr(frame[0], "f_globals"):
                    module_name_from_frame = frame[0].f_globals.get("__name__")

                if (
                    module_name_from_frame
                    and module_name_from_frame.startswith("autogenlib.")
                    and module_name_from_frame != "autogenlib"
                ):
                    # This is an autogenlib module
                    is_autogenlib_exception = True
                    module_name = module_name_from_frame

                    # Get code from cache if it's an autogenlib module
                    if module_name.count(".") > 1:
                        module_name = ".".join(module_name.split(".")[:2])
                    source_code = get_cached_code(module_name)
                    break

                # For non-autogenlib modules, try to read the source file
                try:
                    with open(filename, "r") as f:
                        source_code = f.read()
                    module_name = module_name_from_frame or os.path.basename(
                        filename
                    ).replace(".py", "")
                    break
                except:
                    pass

    # If we couldn't determine the source from the traceback, use the last frame
    if not source_code and tb_frames:
        last_frame = tb_frames[-1]
        if hasattr(last_frame, "filename") and last_frame.filename:
            filename = last_frame.filename
            if (
                "<string>" not in filename
                and filename != "<stdin>"
                and filename.endswith(".py")
            ):
                try:
                    with open(filename, "r") as f:
                        source_code = f.read()
                    module_name = os.path.basename(filename).replace(".py", "")
                except:
                    pass

    # If we still don't have source code but have a module name from an autogenlib module
    if not source_code and module_name and module_name.startswith("autogenlib."):
        source_code = get_cached_code(module_name)
        is_autogenlib_exception = True

    # Check all loaded modules if we still don't have source code
    if not source_code:
        for loaded_module_name, loaded_module in list(sys.modules.items()):
            if (
                loaded_module_name.startswith("autogenlib.")
                and loaded_module_name != "autogenlib"
            ):
                try:
                    # Try to see if this module might be related to the exception
                    if (
                        exc_type.__module__ == loaded_module_name
                        or loaded_module_name in tb_str
                    ):
                        module_name = loaded_module_name
                        if module_name.count(".") > 1:
                            module_name = ".".join(module_name.split(".")[:2])
                        source_code = get_cached_code(module_name)
                        is_autogenlib_exception = True
                        break
                except:
                    continue

    # If we still don't have any source code, try to extract it from any file mentioned in the traceback
    if not source_code:
        for line in tb_str.split("\n"):
            if 'File "' in line and '.py"' in line:
                try:
                    file_path = line.split('File "')[1].split('"')[0]
                    if os.path.exists(file_path) and file_path.endswith(".py"):
                        with open(file_path, "r") as f:
                            source_code = f.read()
                        module_name = os.path.basename(file_path).replace(".py", "")
                        source_file = file_path
                        break
                except:
                    continue

    # If we still don't have source code, we'll just use the traceback
    if not source_code:
        source_code = "# Source code could not be determined"
        module_name = "unknown"

    # Generate fix using LLM
    fix_info = generate_fix(
        module_name,
        source_code,
        exc_type,
        exc_value,
        tb_str,
        is_autogenlib_exception,
        source_file,
    )

    if fix_info and is_autogenlib_exception:
        # For autogenlib modules, we can try to reload them automatically
        fixed_code = fix_info.get("fixed_code")
        if fixed_code:
            # Cache the fixed code
            cache_module(module_name, fixed_code, description)

            # Update the module context
            set_module_context(module_name, fixed_code)

            # Reload the module with the fixed code
            try:
                if module_name in sys.modules:
                    # Execute the new code in the module's namespace
                    exec(fixed_code, sys.modules[module_name].__dict__)
                    logger.info(f"Module {module_name} has been fixed and reloaded")

                    # Output a helpful message to the user
                    print("\n" + "=" * 80)
                    print(f"AutoGenLib fixed an error in module {module_name}")
                    print("The module has been reloaded with the fix.")
                    print("Please retry your operation.")
                    print("=" * 80 + "\n")
            except Exception as e:
                logger.error(f"Error reloading fixed module: {e}")
                print("\n" + "=" * 80)
                print(f"AutoGenLib attempted to fix an error in module {module_name}")
                print(f"But encountered an error while reloading: {e}")
                print("Please restart your application to apply the fix.")
                print("=" * 80 + "\n")
    elif fix_info:
        # For external code, just display the fix suggestions
        print("\n" + "=" * 80)
        print(f"AutoGenLib detected an error in {module_name}")
        if source_file:
            print(f"File: {source_file}")
        print(f"Error: {exc_type.__name__}: {exc_value}")

        # Display the fix suggestions
        print("\nFix Suggestions:")
        print("-" * 40)
        if "explanation" in fix_info:
            explanation = textwrap.fill(fix_info["explanation"], width=78)
            print(explanation)
            print("-" * 40)

        if "fixed_code" in fix_info:
            print("Suggested fixed code:")
            print("-" * 40)
            if source_file:
                print(f"# Apply this fix to {source_file}")

            # If we have specific changes, display them in a more readable format
            if "changes" in fix_info:
                for change in fix_info["changes"]:
                    print(
                        f"Line {change.get('line', '?')}: {change.get('description', '')}"
                    )
                    if "original" in change and "new" in change:
                        print(f"Original: {change['original']}")
                        print(f"New:      {change['new']}")
                        print()
            else:
                # Otherwise just print a snippet of the fixed code (first 20 lines)
                fixed_code_lines = fix_info["fixed_code"].split("\n")
                if len(fixed_code_lines) > 20:
                    print("\n".join(fixed_code_lines[:20]))
                    print("... (truncated for readability)")
                else:
                    print(fix_info["fixed_code"])

        print("=" * 80 + "\n")


def generate_fix(
    module_name,
    current_code,
    exc_type,
    exc_value,
    traceback_str,
    is_autogenlib=False,
    source_file=None,
):
    """Generate a fix for the exception using the LLM.

    Args:
        module_name: Name of the module where the exception occurred
        current_code: Current source code of the module
        exc_type: Exception type
        exc_value: Exception value
        traceback_str: Formatted traceback string
        is_autogenlib: Whether this is an autogenlib-generated module
        source_file: Path to the source file (for non-autogenlib modules)

    Returns:
        Dictionary containing fix information:
        - fixed_code: The fixed code (if available)
        - explanation: Explanation of the issue and fix
        - changes: List of specific changes made (if available)
    """
    try:
        # Set API key from environment variable
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("Please set the OPENAI_API_KEY environment variable.")
            return None

        base_url = os.environ.get("OPENAI_API_BASE_URL")
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1")

        # Initialize the OpenAI client
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        # Create a prompt for the LLM
        prompt = f"""
        You are debugging and fixing a Python error in {
            "code automatically generated by AutoGenLib"
            if is_autogenlib
            else "user code"
        }.

        MODULE DETAILS:
        - Module name: {module_name}
        {f"- Source file: {source_file}" if source_file else ""}

        CURRENT CODE:
        ```python
        {current_code}
        ```

        EXCEPTION DETAILS:
        - Exception type: {exc_type.__name__}
        - Exception message: {exc_value}

        TRACEBACK:
        {traceback_str}

        YOUR TASK:
        {
            "Analyze the error and provide a COMPLETE fixed version of the entire module. Ensure your fix addresses the root cause while maintaining all existing functionality."
            if is_autogenlib
            else "Analyze the error and provide: 1) A clear explanation of what caused the error, 2) Specific line-by-line fix suggestions, and 3) A completely fixed version of the code."
        }

        IMPORTANT GUIDELINES:
        1. Make the minimal changes necessary to fix the issue
        2. Ensure the fix is robust and won't break in edge cases
        3. Maintain the original code's style and approach
        4. Add comments explaining complex fixes if necessary
        5. Check for and fix any other potential issues of similar nature
        6. Ensure type consistency and proper error handling

        {
            "Return ONLY the complete fixed Python code without any explanations or markdown formatting."
            if is_autogenlib
            else f'Format your response as a JSON object with the following structure: {{ "explanation": "brief explanation of the error", "changes": [{{ "line": line_number, "description": "what to change", "original": "original code", "new": "new code" }}], "fixed_code": "complete fixed code" }}'
        }
        """

        # Call the OpenAI API
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Python developer and debugger specialized in fixing code errors. "
                                "Your goal is to analyze errors precisely, identify root causes, and provide robust fixes "
                                "that address both the immediate issue and prevent similar problems.\n\n"
                                "When analyzing errors:\n"
                                "- Trace through the execution flow to find exactly where things went wrong\n"
                                "- Consider edge cases that might have triggered the exception\n"
                                "- Look for patterns of similar issues elsewhere in the code\n"
                                "- Identify underlying design issues that may have contributed\n\n"
                                "When creating fixes:\n"
                                "- Make minimal changes required to fix the issue\n"
                                "- Maintain consistency with the codebase's style and approach\n"
                                "- Add defensive programming when appropriate\n"
                                "- Ensure your fix works for all possible inputs\n"
                                "- Add comments explaining 'why' for non-obvious fixes\n\n"
                                "Provide clear, direct solutions without unnecessary explanations or apologies."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=5000,
                    temperature=0.3,  # Lower temperature for more deterministic results
                    response_format={"type": "json_object"}
                    if not is_autogenlib
                    else None,
                )

                # Get the generated response
                content = response.choices[0].message.content.strip()

                if is_autogenlib:
                    # For autogenlib modules, we expect just the fixed code
                    fixed_code = content

                    # Remove markdown code blocks if present
                    if fixed_code.startswith("```python"):
                        fixed_code = fixed_code.replace("```python", "", 1)
                        fixed_code = fixed_code.replace("```", "", 1)
                    elif fixed_code.startswith("```"):
                        fixed_code = fixed_code.replace("```", "", 1)
                        fixed_code = fixed_code.replace("```", "", 1)

                    fixed_code = fixed_code.strip()

                    # Validate the fixed code
                    try:
                        compile(fixed_code, "<string>", "exec")
                        return {"fixed_code": fixed_code}
                    except Exception as e:
                        logger.warning(f"Generated fix contains syntax errors: {e}")
                        if attempt == max_retries - 1:
                            return None
                        time.sleep(1)  # Wait before retry
                else:
                    # For regular code, we expect a JSON response
                    try:
                        import json

                        fix_info = json.loads(content)

                        # Validate that we have at least some of the expected fields
                        if not any(
                            field in fix_info
                            for field in ["explanation", "changes", "fixed_code"]
                        ):
                            raise ValueError("Missing required fields in response")

                        # If we have fixed code, validate it
                        if "fixed_code" in fix_info:
                            try:
                                compile(fix_info["fixed_code"], "<string>", "exec")
                            except Exception as e:
                                logger.warning(
                                    f"Generated fix contains syntax errors: {e}"
                                )
                                # We'll still return it for user information, even if it has syntax errors

                        return fix_info
                    except Exception as e:
                        logger.warning(f"Error parsing LLM response as JSON: {e}")
                        if attempt == max_retries - 1:
                            # If all attempts failed to parse as JSON, return a simplified response
                            return {
                                "explanation": "Error analyzing the code. Here's the raw LLM output:",
                                "fixed_code": content,
                            }
                        time.sleep(1)  # Wait before retry

            except Exception as e:
                logger.error(f"Error generating fix: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)  # Wait before retry

        return None
    except Exception as e:
        logger.error(f"Error in generate_fix: {e}")
        return None

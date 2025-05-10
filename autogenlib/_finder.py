"""Import hook implementation for autogenlib."""

import sys
import importlib.abc
import importlib.machinery
from ._state import description
from ._generator import generate_code
from ._cache import get_cached_code, cache_module
from ._context import get_module_context, set_module_context


class AutoLibFinder(importlib.abc.MetaPathFinder):
    def __init__(self, desc=None):
        self.description = desc

    def find_spec(self, fullname, path, target=None):
        # Only handle imports under the 'autogenlib' namespace, excluding autogenlib itself
        if not fullname.startswith("autogenlib.") or fullname == "autogenlib":
            return None

        # Get the current description (from instance or global state)
        current_description = self.description or description

        if not current_description:
            return None

        # Determine if this is an attribute import (e.g., autogenlib.totp.function_name)
        is_attribute = fullname.count(".") > 1
        module_name = ".".join(fullname.split(".")[:2])  # e.g., 'autogenlib.totp'

        # Check if the module already exists but is missing this attribute
        if is_attribute and module_name in sys.modules:
            # Split the fullname to get module and attribute
            attr_name = fullname.split(".")[-1]
            module = sys.modules[module_name]

            # If the attribute doesn't exist, regenerate the module
            if not hasattr(module, attr_name):
                # Get the current module code
                module_context = get_module_context(module_name)
                current_code = module_context.get("code", "")

                # Generate updated code including the new function
                new_code = generate_code(current_description, fullname, current_code)
                if new_code:
                    # Update the cache and module
                    cache_module(module_name, new_code, current_description)
                    set_module_context(module_name, new_code)

                    # Execute the new code in the module's namespace
                    exec(new_code, module.__dict__)

                    # If the attribute exists now, return None to continue normal import
                    if hasattr(module, attr_name):
                        return None

        # Is this a package (e.g., autogenlib.totp) or a module?
        is_package = fullname.count(".") == 1

        # Check if the module is already cached
        module_to_check = module_name if is_attribute else fullname
        code = get_cached_code(module_to_check)

        if code is None:
            # Generate code using OpenAI's API
            code = generate_code(current_description, fullname)
            if code is not None:
                # Cache the generated code with the prompt
                cache_module(module_to_check, code, current_description)
                # Update the module context
                set_module_context(module_to_check, code)

        if code is not None:
            # Create a spec for the module
            loader = AutoLibLoader(module_to_check, code)
            return importlib.machinery.ModuleSpec(
                module_to_check, loader, is_package=is_package
            )

        return None


class AutoLibLoader(importlib.abc.Loader):
    def __init__(self, fullname, code):
        self.fullname = fullname
        self.code = code

    def create_module(self, spec):
        return None  # Use the default module creation

    def exec_module(self, module):
        # Execute the generated code in the module's namespace
        exec(self.code, module.__dict__)

        # Update the module context
        set_module_context(self.fullname, self.code)

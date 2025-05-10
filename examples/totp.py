from autogenlib import init

# Initialize with our general description
init("Library for cryptographic operations and secure communications")

# First import - generates the totp module with totp_generator and current_totp_code functions
from autogenlib.totp import generate_totp_secret, current_totp_code

secret = generate_totp_secret()
print(f"Secret: {secret}")

code = current_totp_code(secret)
print(f"Code: {code}")

from autogenlib.totp import validate_code_against_secret

print(f"Is valid?: {validate_code_against_secret(code, secret)}")

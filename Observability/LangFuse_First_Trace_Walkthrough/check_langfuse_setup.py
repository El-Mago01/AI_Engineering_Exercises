
"""
Before running the traced app, check that the LangFuse values can be loaded from .env.

If any value says MISSING, go back to the previous page and check the .env file.
If all values say OK, the app can send traces to your LangFuse project.
"""

import os

from dotenv import load_dotenv


load_dotenv()


REQUIRED = [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
]


for name in REQUIRED:
    value = os.getenv(name)
    print(name, "OK" if value else "MISSING")

def main():
    print("LANGFUSE SETUP CHECK")
    print("--------------------")

    missing = []
    for name in REQUIRED:
        value = os.getenv(name)
        status = "OK" if value else "MISSING"
        print(f"{name}: {status}")
        if not value:
            missing.append(name)

    if missing:
        print("\nAdd the missing LangFuse values to .env before expecting traces to appear.")
        return

    print("\nLangFuse values loaded from .env.")


if __name__ == "__main__":
    main()

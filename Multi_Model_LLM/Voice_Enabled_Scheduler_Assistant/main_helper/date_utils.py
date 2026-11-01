"""
main_helper/date_utils.py: Provides utility functions for handling dates.
"""

from datetime import datetime

def format_date(date):
    suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
    if 10 <= date.day <= 20 or date.day % 10 not in suffixes:
        suffix = 'th'
    else:
        suffix = suffixes[date.day % 10]
    return date.strftime(f"%B {date.day}{suffix}, %Y")

# Example usage
date = datetime(2024, 6, 21)
print(format_date(date))  # Output: June 21st, 2024
import sys
from typing import Any
from src.logger import logging

"""
Whenever we create an object of CustomException, Python first calls its __init__ method, which internally invokes super().__init__(error_message) to construct a standard Python Exception object using the inherited base class. This ensures that our custom exception behaves like any built-in error — it can be raised, caught, and logged. The constructor of Exception expects a basic error message, which we provide. Then, we enrich that error by calling a helper function that extracts traceback details like filename and line number, and we store this formatted message as an attribute (self.error_message) inside our custom object. Finally, we override the __str__() method so that printing the exception shows our enhanced message instead of the default one. In conclusion, CustomException creates a fully functional Python exception object with additional diagnostic configuration, making it more informative and traceable for debugging and automation workflows

"""

def error_message_details(error : str, error_detail: Any) -> str:
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = f"Error occurred in script: {file_name} at line number: {line_number} with message: {error}"
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Any)-> None: 
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)
        
    def __str__(self):
        return self.error_message

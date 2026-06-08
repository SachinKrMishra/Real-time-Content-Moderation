import sys
from src.logging import logger

class ModerationException(Exception):
    def __init__(self, error_message, error_details:sys):
        self.error_message = error_message
        _,_,exc_tb = error_details.exc_info()  ## exc_tb -> gives the entire error detail

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return "Error occurred in Python Script name [{0}] line number [{1}] error_message [{2}]".format(self.file_name, self.lineno, str(self.error_message))
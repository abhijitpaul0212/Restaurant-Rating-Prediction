# exception.py

import sys


class CustomException(Exception):

    def __init__(self, error_message: Exception, error_details: sys) -> None:
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()

        self.line_no = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self) -> str:
        return f"File name [{self.file_name}] Line number [{self.line_no}] Error Message [{str(self.error_message)}]"


if __name__ == '__main__':
    try:
        print(1/0)
    except Exception as e:
        raise CustomException(e, sys)

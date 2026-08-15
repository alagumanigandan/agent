#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from stock_analyst.crew import StockAnalyst

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information
import os
os.makedirs("./op", exist_ok=True)
def run():
    """
    Run the crew.
    """
    inputs = {
        "company_name": "Tesla"
    }

   
    result= StockAnalyst().crew().kickoff(inputs=inputs)
    
if __name__ == "__main__":
    run()



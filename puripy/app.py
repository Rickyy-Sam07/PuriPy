import logging
import sys
import os

# Configure logging properly
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleansipy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clear_screen():
    """Clear the terminal screen based on OS"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("Welcome to CleansiPy")
    
    response = input(
                "What kind of data cleaning you want \n"
                "1. press 1 for numeric data  \n"
                "2. press 2 for categorical data \n"
                "3. press 3 for textual data cleaning \n"
                "4. press 4 for date & time data cleaning \n"
                "5. press 5 for exit \n"
                "Enter your choice: "
                ).strip().lower()
    
    # Check for exit first
    if response == "5":
        print("Exiting CleansiPy. Goodbye!")
        return
    
    # Check if the response is valid
    if response not in ["1", "2", "3", "4"]:
        print("Invalid option. Please run the program again and select options 1-5.")
        return
    
    y = input("Have you set the config in config.py (y/n): ").strip().lower()
    is_config_ready = y in ["yes", "y", "YES", "Yes"]
    
    if not is_config_ready:
        print("Please set the config in config.py before proceeding.")
        logger.warning("User did not set config in config.py")
        return
    
    try:
        # Clear the screen to avoid console output interference
        clear_screen()
        print(f"Starting processing for option {response}...\n")
        
        if response == "1":
            from .mainnum import main2
            main2() 
        elif response == "2":
            from .maincat import main0
            main0()
        elif response == "3":
            from .maintext import main as main_text
            main_text()
        elif response == "4":
            from .maindt import main3
            main3()
            
        print("\nProcessing complete. Thank you for using CleansiPy!")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        print("Make sure you have set the config correctly in config.py")

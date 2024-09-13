import os



def check_directory_contents(directory: str):
    """
    Checks if the specified directory exists and contains files or subdirectories.

    Args:
        directory (str): Path to the directory to check.

    Returns:
        bool: True if directory exists and has contents, False otherwise.
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return False
    
    # Check if the directory contains any files or subdirectories
    if not os.listdir(directory):
        print(f"The directory {directory} is empty.")
        return False

    # If there are files or subdirectories
    print(f"The directory {directory} exists and has contents.")
    return True

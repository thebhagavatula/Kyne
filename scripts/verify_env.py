import sys

def verify_environment() -> None:
    """
    Verifies that the required environment packages cv2, numpy, and dtaidistance are installed.
    
    Parameters:
        None
        
    Returns:
        None
    """
    missing_packages = False


    try:
        import cv2
        print(f"cv2 version: {cv2.__version__}")
    except ImportError as e:
        print(f"Error importing cv2: {e}")
        print("Please ensure opencv-python is installed.")
        missing_packages = True

    try:
        import numpy as np
        print(f"numpy version: {np.__version__}")
    except ImportError as e:
        print(f"Error importing numpy: {e}")
        missing_packages = True

    try:
        import dtaidistance
        print(f"dtaidistance version: {dtaidistance.__version__}")
    except ImportError as e:
        print(f"Error importing dtaidistance: {e}")
        missing_packages = True

    if missing_packages:
        print("\nEnvironment verification failed. Please install the missing packages.")
        sys.exit(1)
        
    print("\nEnvironment verification passed successfully!")

if __name__ == "__main__":
    verify_environment()

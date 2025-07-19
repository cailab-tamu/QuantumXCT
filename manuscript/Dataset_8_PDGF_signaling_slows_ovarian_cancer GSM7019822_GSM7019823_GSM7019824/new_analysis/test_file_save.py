import os
import json

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, 'test_output.json')

try:
    # Test data
    test_data = {
        'test': 'This is a test file',
        'timestamp': '2025-07-19T09:52:15-05:00'
    }
    
    # Try to save the file
    with open(output_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("="*60)
    print("TEST FILE SAVE SUCCESSFUL")
    print("="*60)
    print(f"File saved to: {output_file}")
    
    # Verify the file was written
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size} bytes")
        
        # Read back the file to verify contents
        with open(output_file, 'r') as f:
            content = f.read()
            print("\nFile contents:")
            print("-"*30)
            print(content)
            print("-"*30)
    else:
        print("ERROR: File was not created!")
        
except Exception as e:
    print("\n" + "!"*60)
    print("ERROR: Failed to save test file:")
    print(str(e))
    print("Current working directory:", os.getcwd())
    print("Target directory exists:", os.path.exists(script_dir))
    print("Target directory is writable:", os.access(script_dir, os.W_OK))
    print("!"*60 + "\n")

import codecs
import os # Import the os module for file path operations

def decode_unicode_escapes(input_string):
  """
  Converts Unicode escape sequences (e.g., '\\u00e8') in a string
  to their corresponding actual characters (e.g., 'è').

  Args:
    input_string: The string containing Unicode escape sequences.

  Returns:
    The string with escape sequences converted to actual characters,
    or None if an error occurs during decoding.
  """
  try:
    # The 'unicode_escape' codec specifically handles these escape sequences.
    # It interprets patterns like \uXXXX and \UXXXXXXXX.
    decoded_string = codecs.decode(input_string, 'unicode_escape')
    return decoded_string
  except Exception as e:
    print(f"Error during decoding: {e}")
    return None

def process_file(input_filepath, output_filepath=None):
  """
  Reads a file, decodes Unicode escape sequences in its content,
  and prints or saves the result.

  Args:
    input_filepath: Path to the input file.
    output_filepath: Optional path to save the converted content.
                     If None, the content is printed to the console.
  """
  try:
    # Open and read the input file
    # Using 'utf-8' encoding is generally a good practice, though
    # 'unicode_escape' primarily works on the escape sequences themselves.
    with open(input_filepath, 'r', encoding='utf-8') as infile:
      content_with_escapes = infile.read()
      print(f"--- Original content from {input_filepath} ---")
      print(content_with_escapes)
      print("-" * (len(input_filepath) + 28)) # Separator line

      # Convert the content
      converted_content = decode_unicode_escapes(content_with_escapes)

      if converted_content is not None:
        if output_filepath:
          # Write the converted content to the output file
          with open(output_filepath, 'w', encoding='utf-8') as outfile:
            outfile.write(converted_content)
          print(f"\n--- Converted content saved to {output_filepath} ---")
        else:
          # Print the converted content to the console
          print("\n--- Converted content ---")
          print(converted_content)
          print("-" * 25) # Separator line

  except FileNotFoundError:
    print(f"Error: Input file not found at '{input_filepath}'")
  except IOError as e:
    print(f"Error reading or writing file: {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")

# --- Configuration ---
# Define the input filename
input_filename = "det_00041_15-01-2025.json"
# Define an optional output filename (set to None to just print)
output_filename = "test_converted.txt"
# output_filename = None # Uncomment this line to print instead of saving

# --- Execution ---
# Construct the full path to the input file (optional, but good practice)
# Assumes the file is in the same directory as the script
input_file_path = os.path.join(os.path.dirname(__file__), input_filename) if '__file__' in locals() else input_filename

# Before running, create a dummy 'test.json' file in the same directory
# with content like: "Questo \\u00e8 un test con caratteri speciali: \\u00e0 \\u00e9 \\u00f2 \\u00f9"

# Check if the dummy file exists, if not, create it for demonstration
if not os.path.exists(input_file_path):
    print(f"Creating dummy file: {input_file_path}")
    try:
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write("Questo \\u00e8 un test con caratteri speciali: \\u00e0 \\u00e9 \\u00f2 \\u00f9 and Euro: \\u20ac")
        print("Dummy file created successfully.")
    except IOError as e:
        print(f"Could not create dummy file: {e}")
        # Exit if we can't create the file for the demo
        exit()


# Process the file
process_file(input_file_path, output_filename)


# --- Original Example Usage (Commented out) ---
# string_with_escapes = "H\\u0065llo, this is a test string with special characters like \\u00e8, \\u00e9, \\u00e0 and the Euro sign \\u20ac."
# print(f"Original string: {string_with_escapes}")
# converted_string = decode_unicode_escapes(string_with_escapes)
# print(f"Converted string: {converted_string}")
# another_example = "Perch\\u00e9 non funziona?"
# print(f"\nOriginal string: {another_example}")
# converted_example = decode_unicode_escapes(another_example)
# print(f"Converted string: {converted_example}")

import os
import json

def update_json_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                # Read the JSON file
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Ensure "Responses" is a list and update each element
                    if "Responses" in data and isinstance(data["Responses"], list):
                        for response in data["Responses"]:
                            response["model"] = "gpt-4o-mini"

                        # Write back the updated JSON
                        with open(file_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=4)
                    
                    print(f"Updated: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Run the function on the current directory (or specify a different path)
update_json_files(".")
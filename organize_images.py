import os

def organize_and_rename_images(directory):
    # Get a list of all files in the directory
    files = os.listdir(directory)
    
    # Filter out only the PNG files (you can change this to include other formats if needed)
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    # Sort the files by their original names
    png_files.sort()

    # Rename each file sequentially
    for index, file_name in enumerate(png_files, start=1):
        old_path = os.path.join(directory, file_name)
        new_file_name = f"{index}.png"
        new_path = os.path.join(directory, new_file_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_file_name}")

if __name__ == "__main__":
    # Set your directory path here
    directory_path = "captured_images"
    organize_and_rename_images(directory_path)

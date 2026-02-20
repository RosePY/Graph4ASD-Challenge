import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

class NoEncryptedFileError(Exception):
    pass

from leaderboard.update_leaderboard import update_leaderboard_csv

SUBMISSION_DIR = os.path.join(project_root, "submissions")

def read_latest_submission():
    changed_files_str = os.getenv("CHANGED_FILES", "").strip()
    
    if not changed_files_str:
        print("No files provided in CHANGED_FILES environment variable.")
        # Return empty list instead of looking at the whole directory
        return []

    # Filter and create absolute paths
    files = [
        os.path.abspath(os.path.join(project_root, f.strip())) 
        for f in changed_files_str.split() 
        if f.endswith(".enc")
    ]
    return files
    
# def read_latest_submission():
#     # Try to get files from the git diff provided by the workflow
#     changed_files_str = os.getenv("CHANGED_FILES", "")
#     print("changed_files_str",changed_files_str)
#     if changed_files_str:
#         # Split by newline/space and get the first .enc file
#         files = [f.strip() for f in changed_files_str.split() if f.endswith(".enc")]
#         print("files", files) 
#         if files:
#             return os.path.abspath(files[0])
            
#     files = [f for f in os.listdir(SUBMISSION_DIR) if f.endswith(".enc")]
#     if not files:
#         raise NoEncryptedFileError("No encrypted submission files found in 'submissions' directory.")
#     latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(SUBMISSION_DIR, f)))
#     print(latest_file)
#     return os.path.join(SUBMISSION_DIR, latest_file)


def decrypt_submission_file(encrypted_file_path):
    from encryption.decrypt import decrypt_file_content
    decrypted_content = decrypt_file_content(encrypted_file_path)
    decrypted_file_path = encrypted_file_path.replace(".enc", "")
    with open(decrypted_file_path, "wb") as f:
        f.write(decrypted_content)
    return decrypted_file_path

def calculate_submission_score(decrypted_file_path):
    from leaderboard.calculate_scores import calculate_scores
    score = calculate_scores(decrypted_file_path)
    return score

def process_submission():
    try:
        encrypted_file = read_latest_submission()
        decrypted_file = decrypt_submission_file(encrypted_file)
        # now the data is saved to a decrypted file ending with ".csv"
        # the leaderboard should be set up to automatically pick up this file and use the logic in calculate_scores.py
        # to show the new entries (files ending with .csv) on the leaderboard
        update_leaderboard_csv()
    except NoEncryptedFileError as e:
        print(f"Error: {e}")
        print("Continuing with no changes...")

if __name__ == "__main__":
    process_submission()

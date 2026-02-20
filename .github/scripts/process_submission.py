import os
import sys

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(project_root)

class NoEncryptedFileError(Exception):
    pass

from leaderboard.update_leaderboard import update_leaderboard_csv
from encryption.decrypt import decrypt_file_content
from leaderboard.calculate_scores import calculate_scores

SUBMISSION_DIR = os.path.join(project_root, "submissions")
print("SUBMISSIONS DIR CONTENT:")
print(os.listdir(SUBMISSION_DIR))

def read_latest_submission():
    changed_files_str = os.getenv("CHANGED_FILES", "").strip()
    
    if not changed_files_str:
        print("No files provided in CHANGED_FILES environment variable.")
        return []

    # Filter and create absolute paths
    files = [
        os.path.abspath(os.path.join(project_root, f.strip())) 
        for f in changed_files_str.split() 
        if f.endswith(".enc")
    ]
    return files

def decrypt_submission_file(encrypted_file_path):
    # Pass the path directly to the decryption logic
    decrypted_content = decrypt_file_content(encrypted_file_path)
    decrypted_file_path = encrypted_file_path.replace(".enc", "")
    with open(decrypted_file_path, "wb") as f:
        f.write(decrypted_content)
    return decrypted_file_path

def process_submission():
    summary_lines = ["### Submission Results", ""]
    processed_any = False
    
    try:
        encrypted_files = read_latest_submission()
        
        if not encrypted_files:
            summary_lines.append("No new `.enc` files detected for processing.")
        else:
            # LOOP through the list of files
            for enc_file in encrypted_files:
                try:
                    print(f"Processing: {enc_file}")
                    
                    # 1. Decrypt
                    decrypted_file = decrypt_submission_file(enc_file)
                    
                    # 2. Score (for the summary comment)
                    score = calculate_scores(decrypted_file)
                    
                    summary_lines.append(f" **File:** `{os.path.basename(enc_file)}`")
                    summary_lines.append(f" **Score:** `{score}`")
                    summary_lines.append("---")
                    processed_any = True
                    
                except Exception as e:
                    summary_lines.append(f"**File:** `{os.path.basename(enc_file)}`")
                    summary_lines.append(f"**Error:** {str(e)}")
                    summary_lines.append("---")

        update_leaderboard_csv()
        print("Leaderboard actualizado con éxito.")
        
    except Exception as e:
        print(f"Critical error: {e}")
        summary_lines.append(f"**Critical Error:** {str(e)}")

    # Write the summary for the GitHub Action to post as a comment
    with open(os.path.join(project_root, "submission_summary.md"), "w") as f:
        f.write("\n".join(summary_lines))

if __name__ == "__main__":
    process_submission()

# from glob import glob
# import os
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# files = glob("./out/huggingface/*")
#
# print("Available files:")
# for i, path in enumerate(files):
#     print(f"{i+1}. {path}")
#
# choice = input("Choose a file number to convert, or type 'all' to convert all files: ")
#
# if choice.lower() == 'all':
#     files_to_process = files
# else:
#     try:
#         file_index = int(choice) - 1
#         if 0 <= file_index < len(files):
#             files_to_process = [files[file_index]]
#         else:
#             print("Invalid file number. No files will be converted.")
#             files_to_process = []
#     except ValueError:
#         print("Invalid input. Please type 'all' or a file number. No files will be converted.")
#         files_to_process = []
#
# ckpts = glob("./checkpoints/*/*")
# print("Available Models:")
# for i, path in enumerate(ckpts):
#     print(f"{i+1}. {path}")
#
# ckpt_choice = input("Please select the respective model: ")
# try:
# 	if 0 < int(ckpt_choice) <= len(ckpts):
# 		f = ckpts[int(ckpt_choice) - 1]
# 		for path in files_to_process:
# 			if not os.path.exists(path):
# 				continue
# 			state_dict = torch.load(path+"/model.pth", weights_only=True)
# 			model = AutoModelForCausalLM.from_pretrained(
# 					f,
# 					torch_dtype="auto",
# 					device_map="cuda:0",
# 					local_files_only=True
# 				)
# 			model.load_state_dict(state_dict)
# 			tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
# 			model.save_pretrained(path.replace("huggingface", "safetensors"))
# 			tokenizer.save_pretrained(path.replace("huggingface", "safetensors"))
# 			del model, tokenizer, state_dict
# except ValueError:
#         print("Invalid input. Please type a file number. No files will be converted.")


from glob import glob
import os
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Convert model files to safetensors format')
    parser.add_argument('--input_dir', '-i', type=str, default='./out/huggingface',
                        help='Directory containing input files')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='Directory for output files (defaults to input_dir with "huggingface" replaced by "safetensors")')
    parser.add_argument('--checkpoint_dir', '-c', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--file', '-f', type=str, default='all',
                        help='Specific file to convert (number or name) or "all" for all files')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Override automatic model selection (number or path)')
    parser.add_argument('--list_only', '-l', action='store_true',
                        help='List available files and models without converting')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to use for conversion (e.g., "cuda:0", "cpu")')
    return parser.parse_args()


def list_files(directory, pattern="*"):
    files = glob(os.path.join(directory, pattern))
    print(f"Available files in {directory}:")
    for i, path in enumerate(files):
        print(f"{i + 1}. {path}")
    return files


def get_files_to_process(args, input_files):
    if args.file.lower() == 'all':
        return input_files

    # Try as index first, then as path match
    try:
        file_index = int(args.file) - 1
        if 0 <= file_index < len(input_files):
            return [input_files[file_index]]
    except ValueError:
        matching_files = [f for f in input_files if args.file in f]
        if matching_files:
            return matching_files

    print(f"No matching file found for '{args.file}'")
    return []


def get_model_name_from_path(path):
    """Extract model name from file path for matching with checkpoints"""
    basename = os.path.basename(path).lower()
    segments = basename.split('-')

    # Common model family names to look for
    model_families = ['qwen', 'llama', 'gemma', 'smollm', 'mistral', 'falcon', 'phi']

    # Try to identify model family and version
    for segment in segments:
        for family in model_families:
            if family in segment.lower():
                return family

    # If no match found, just return the base folder name
    return basename


def auto_match_checkpoint(input_path, checkpoints):
    """Automatically match input model to most appropriate checkpoint"""
    model_name = get_model_name_from_path(input_path)
    print(f"Detected model family: {model_name}")

    # Find matching checkpoints by model name
    matches = []
    for ckpt in checkpoints:
        if model_name.lower() in ckpt.lower():
            matches.append(ckpt)

    # Look for size/parameter matches if available
    if matches:
        # Check for specific size identifiers
        size_indicators = ["3b", "2b", "1.5b", "1b", "0.5b", "135m", "360m", "1.7b"]
        input_path_lower = input_path.lower()

        for indicator in size_indicators:
            if indicator in input_path_lower:
                for match in matches:
                    if indicator in match.lower():
                        print(f"Auto-selected checkpoint: {match} (size match: {indicator})")
                        return match

        # If no size match, return the first model family match
        print(f"Auto-selected checkpoint: {matches[0]} (family match)")
        return matches[0]

    # If no match found, return None
    print(f"No matching checkpoint found for {model_name}")
    return None


def select_checkpoint(args, input_path, checkpoints):
    # If manual override provided, use that
    if args.model:
        # Try as index first, then as direct path, then as partial match
        try:
            ckpt_index = int(args.model) - 1
            if 0 <= ckpt_index < len(checkpoints):
                return checkpoints[ckpt_index]
        except ValueError:
            if os.path.exists(args.model):
                return args.model

            matching_ckpts = [c for c in checkpoints if args.model in c]
            if matching_ckpts:
                return matching_ckpts[0]

        print(f"No matching checkpoint found for '{args.model}'")
        return None

    # Try automatic selection first
    auto_ckpt = auto_match_checkpoint(input_path, checkpoints)
    if auto_ckpt:
        return auto_ckpt

    # Fall back to interactive selection if automatic selection fails
    print("\nAutomatic selection failed. Please choose a checkpoint manually.")
    print("\nAvailable Models:")
    for i, path in enumerate(checkpoints):
        print(f"{i + 1}. {path}")

    while True:
        try:
            choice = input("Select model number: ")
            ckpt_index = int(choice) - 1
            if 0 <= ckpt_index < len(checkpoints):
                return checkpoints[ckpt_index]
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(checkpoints)}.")
        except ValueError:
            print("Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def convert_model(input_path, output_path, checkpoint_path, device):
    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return False

    try:
        print(f"Converting {input_path} → {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load state dict and model efficiently
        state_dict = torch.load(os.path.join(input_path, "model.pth"), weights_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype="auto",
            device_map=device,
            local_files_only=True
        )
        model.load_state_dict(state_dict)
        tokenizer = AutoTokenizer.from_pretrained(input_path, local_files_only=True)

        # Save in safetensors format
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Clean up memory
        torch.cuda.empty_cache()  # Explicitly clear CUDA cache
        del model, tokenizer, state_dict

        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    args = parse_args()

    # Get files and checkpoints
    input_files = list_files(args.input_dir)  # Changed to use list_files to show options
    checkpoints = glob(os.path.join(args.checkpoint_dir, "*", "*"))

    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    if not input_files:
        print(f"No input files found in {args.input_dir}")
        return

    # Show checkpoint options too
    print("\nAvailable Models:")
    for i, path in enumerate(checkpoints):
        print(f"{i + 1}. {path}")

    # Get user choice for files
    while True:
        choice = input("\nChoose a file number to convert, or type 'all' to convert all files: ")

        if choice.lower() == 'all':
            files_to_process = input_files
            break
        else:
            try:
                file_index = int(choice) - 1
                if 0 <= file_index < len(input_files):
                    files_to_process = [input_files[file_index]]
                    break
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(input_files)}.")
            except ValueError:
                print("Invalid input. Please type 'all' or a file number.")

    if not files_to_process:
        print("No files to convert.")
        return

    # Convert files with progress tracking
    successful = 0
    total = len(files_to_process)

    for i, input_path in enumerate(files_to_process):
        print(f"\n[{i + 1}/{total}] Processing: {input_path}")

        # Select appropriate checkpoint for this input file
        checkpoint_path = select_checkpoint(args, input_path, checkpoints)
        if not checkpoint_path:
            print("No valid checkpoint. Skipping this file.")
            continue

        # Determine output path
        if args.output_dir:
            output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        else:
            output_path = input_path.replace("huggingface", "safetensors")

        if convert_model(input_path, output_path, checkpoint_path, args.device):
            successful += 1

    print(f"Completed: {successful}/{total} conversions successful")

    '''args = parse_args()

    # Get files and checkpoints
    input_files = glob(os.path.join(args.input_dir, "*"))
    checkpoints = glob(os.path.join(args.checkpoint_dir, "*", "*"))

    if not checkpoints:
        print(f"No checkpoints found in {args.checkpoint_dir}")
        return

    if not input_files:
        print(f"No input files found in {args.input_dir}")
        return

    # Just list if requested
    if args.list_only:
        list_files(args.input_dir)
        print("\nAvailable Models:")
        for i, path in enumerate(checkpoints):
            print(f"{i + 1}. {path}")
        return

    # Otherwise show available options
    print(f"Found {len(input_files)} input files in {args.input_dir}")

    # Process files
    files_to_process = get_files_to_process(args, input_files)
    if not files_to_process:
        print("No files to convert.")
        return

    # Convert files with progress tracking
    successful = 0
    total = len(files_to_process)

    for i, input_path in enumerate(files_to_process):
        print(f"\n[{i + 1}/{total}] Processing: {input_path}")

        # Select appropriate checkpoint for this input file
        checkpoint_path = select_checkpoint(args, input_path, checkpoints)
        if not checkpoint_path:
            print("No valid checkpoint. Skipping this file.")
            continue

        # Determine output path
        if args.output_dir:
            output_path = os.path.join(args.output_dir, os.path.basename(input_path))
        else:
            output_path = input_path.replace("huggingface", "safetensors")

        if convert_model(input_path, output_path, checkpoint_path, args.device):
            successful += 1

    print(f"Completed: {successful}/{total} conversions successful")'''


if __name__ == "__main__":
    main()

import argparse
import subprocess
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def get_args():
    description = """
    This script preprocesses the raw pcap data into the tokenized format. It takes the input folder as an argument and one of two required flags: --pretrain or --finetune.
    The input folder must contain '/raw' folder with either raw pcap files (for pretraining, no labels) or folders with pcap files (finetuning, folder names must be integers and would be used as labels).
    The input folder would be used for intermediate files and the final tokenized data would be stored in the <input_folder>/final/shards folder as Apache Arrow shards.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_folder", type=str, required=True, help="The input folder")
    parser.add_argument("--action", choices=["pretrain", "finetune"], required=True,
                        help="Preprocess data for pretraining or finetuning.")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="The tokenizer config file.")
    parser.add_argument("--tcp_options", action="store_true", default=False, help="Include TCP options in the tokenized data.")
    parser.add_argument("--combined", action="store_true", default=False,
                        help="Combine all the pcap files in the /final/shards into a single file (suitable for small datasets).")

    return parser


def run(command: list[str]) -> subprocess.CompletedProcess:
    logger.info(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, check=True, capture_output=True)
    if process.stderr:
        logger.error(process.stderr.decode())
    if process.stdout:
        logger.info(process.stdout.decode())
    return process


def get_base_directory(args):
    # one step up of the directory of this file
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def preprocess_pretrain(args):
    base_directory = get_base_directory(args)
    input_folder = args.input_folder
    run([f"{base_directory}/pre_process_src/1_filter.sh", f"{input_folder}/raw", f"{input_folder}/filtered"])
    run([f"{base_directory}/pre_process_src/2_pcap_splitting.sh", f"{input_folder}/filtered", f"{input_folder}/split"])
    run([f"{base_directory}/pre_process_src/3_extract_fields.sh", f"{input_folder}/split", f"{input_folder}/extracted", "1" if args.tcp_options else ""])

    for folder_name in os.listdir(f"{input_folder}/extracted"):
        full_folder_name = os.path.join(f"{input_folder}/extracted", folder_name)
        os.makedirs(os.path.join(f"{input_folder}/final/shards", folder_name), exist_ok=True)
        run(["python3", f"{base_directory}/pre_process_src/Tokenize.py", "--conf_file", args.tokenizer_config,
             "--input_dir", full_folder_name, "--output_dir",
             os.path.join(f"{input_folder}/final/shards", folder_name)])
        if args.combined:
            os.makedirs(os.path.join(f"{input_folder}/final", "combined"), exist_ok=True)
            run(["python3", f"{base_directory}/pre_process_src/CollectTokensInFiles.py",
                 os.path.join(f"{input_folder}/final/shards", folder_name),
                 os.path.join(f"{input_folder}/final/combined", f"{folder_name}.arrow")])


def preprocess_finetune(args):
    base_directory = get_base_directory(args)
    input_folder = args.input_folder
    for label in os.listdir(f"{input_folder}/raw"):
        for stage_name in ["filtered", "split", "extracted", "final/shards"]:
            os.makedirs(os.path.join(input_folder, stage_name, label), exist_ok=True)
        run([f"{base_directory}/pre_process_src/1_filter.sh", f"{input_folder}/raw/{label}",
             f"{input_folder}/filtered/{label}"])
        run([f"{base_directory}/pre_process_src/2_pcap_splitting.sh", f"{input_folder}/filtered/{label}",
             f"{input_folder}/split/{label}"])
        run([f"{base_directory}/pre_process_src/3_extract_fields.sh", f"{input_folder}/split/{label}",
             f"{input_folder}/extracted/{label}", "1" if args.tcp_options else ""])

        for folder_name in os.listdir(f"{input_folder}/extracted/{label}"):
            full_folder_name = os.path.join(f"{input_folder}/extracted/{label}", folder_name)
            os.makedirs(os.path.join(f"{input_folder}/final/shards/{label}", folder_name), exist_ok=True)
            run(["python3", f"{base_directory}/pre_process_src/Tokenize.py", "--conf_file", args.tokenizer_config,
                 "--input_dir", full_folder_name, "--output_dir",
                 os.path.join(f"{input_folder}/final/shards/{label}", folder_name), '--label', label])
            if args.combined:
                os.makedirs(os.path.join(f"{input_folder}/final", "combined"), exist_ok=True)
                run(["python3", f"{base_directory}/pre_process_src/CollectTokensInFiles.py",
                     os.path.join(f"{input_folder}/final/shards/{label}", folder_name),
                     os.path.join(f"{input_folder}/final/combined", f"{label}_{folder_name}.arrow")])

def main():
    parser = get_args()
    args = parser.parse_args()
    input_folder = args.input_folder
    action = args.action

    raw_data_folder = os.path.join(input_folder, "raw")
    if not os.path.exists(raw_data_folder):
        print(f"Input folder {raw_data_folder} does not exist.")
        return

    for folder in ["filtered", "split", "extracted", "final", "final/shards"]:
        os.makedirs(os.path.join(input_folder, folder), exist_ok=True)

    match action:
        case "pretrain":
            preprocess_pretrain(args)
        case "finetune":
            preprocess_finetune(args)
        case _:
            raise ValueError("Unexpected action")

    return


if __name__ == "__main__":
    main()

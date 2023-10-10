import argparse
import os, json
import torch


def _save_to_pth(input_path, data):
    fpart, fext = os.path.splitext(input_path)
    pth_path = f"{fpart}.pth"
    torch.save(data, pth_path)

def _load_from(input_path):
    fpart, fext = os.path.splitext(input_path)
    pth_path = f"{fpart}.pth"
    if os.path.isfile(pth_path):
        return
    data = []
    with open(input_path, "r", encoding="utf-8") as fp:
        for index, line in enumerate(fp):
            rec = json.loads(line)
            if "doc" in rec:
                del rec["doc"] # Reduce memory footprint.
            data.append(rec)
    _save_to_pth(input_path, data)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Pickle dataset")
    parser.add_argument('--dataset_root', type=str, required=False, default=".",
                        help='The root directory in which training dataset directory resides.')
    parser.add_argument("--dataset_folder", type=str, default=None,
                        help="The directory where prebuilt datasets are saved and loaded.")
    parser.add_argument("--split_type", type=str, default=None,
                        help="The dataset splits (e.g. train,valid).")
    parser.add_argument("--pair_type", type=str, default=None,
                        help="The dataset pairs (e.g. atcl,hlit).")
    parser.add_argument("--ext_type", type=str, default=None,
                        help="The file extention.")
    parser.add_argument("--dataset_file_format", type=str, default="{split_type}.{pair_type}.{ext_type}",
                        help="A a json file containing the training data.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    dataset_dir = os.path.join(args.dataset_root, args.dataset_folder)
    split_types = args.split_type.split(",")
    pair_types = args.pair_type.split(",")
    for split_type in split_types:
        for pair_type in pair_types:
            fpath = args.dataset_file_format.format(split_type=split_type,
                                                    pair_type=pair_type,
                                                    ext_type=args.ext_type)
            fpath = f"{dataset_dir}/{fpath}"
            print(f"Pickling {fpath}")
            _load_from(fpath)

if __name__ == "__main__":
    main()
    print("Done!")

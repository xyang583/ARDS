import os
import argparse
import json
from llava.data.collect_grad_reps import merge_and_normalize_info, merge_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prefix', type=str, default="grads")
    parser.add_argument('--proj_dim', type=int, nargs='+', default=[8192])
    parser.add_argument('--woproj', action='store_true', default=False)
    parser.add_argument('--save_normalize', action='store_true', default=False)
    return parser.parse_args()

def merge(output_dir, proj_dim, prefix):
    for dim in proj_dim:
        output_dir = os.path.join(output_dir, f"dim{dim}")
        merge_and_normalize_info(output_dir, prefix=prefix)
        merge_info(output_dir, prefix=prefix)

if __name__ == '__main__':

    args = parse_args()
    if not args.woproj:
        merge(args.output_dir, args.proj_dim, prefix=args.prefix)
    else:
        merge_info(args.output_dir, prefix=args.prefix)

        if args.save_normalize:
            merge_and_normalize_info(args.output_dir, prefix=args.prefix)

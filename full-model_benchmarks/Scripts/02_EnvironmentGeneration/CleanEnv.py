import os, glob
import argparse

"""
Cleans the execution environment generated by BuildExecutionEnv.py
If there are results stored in the environment, it will not be cleaned
(unless -f or --force is used)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean the environment')
    parser.add_argument('layer_dir', type=str, help='Directory containing the information of the layer')
    parser.add_argument('-f', '--force', action='store_true', help='Force clean even if there is results stored in the environment')
    args = parser.parse_args()

    # check if there is results stored in the environment
    if not args.force:
        result_files = []
        for dir in ['inner_product_m', 'inner_product_n', 'outer_product_m', 'outer_product_n', 'gustavsons_m', 'gustavsons_n']:
            result_files += glob.glob(os.path.join(args.layer_dir, dir, 'output*'))

        if len(result_files) > 0:
            print('There are results stored in the environment. Please manage them first.')
            print('If you want to force clean the environment, please use -f or --force.')
            exit()

    # delete the environment
    os.system(f"rm -rf {os.path.join(args.layer_dir, 'inner_product_m')}")
    os.system(f"rm -rf {os.path.join(args.layer_dir, 'inner_product_n')}")
    os.system(f"rm -rf {os.path.join(args.layer_dir, 'outer_product_m')}")
    os.system(f"rm -rf {os.path.join(args.layer_dir, 'outer_product_n')}")
    os.system(f"rm -rf {os.path.join(args.layer_dir, 'gustavsons_m')}")
    os.system(f"rm -rf {os.path.join(args.layer_dir, 'gustavsons_n')}")
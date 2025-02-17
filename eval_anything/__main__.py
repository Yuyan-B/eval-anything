import argparse
from eval_anything.utils.utils import parse_unknown_args
from eval_anything.pipeline.base_task import BaseTask
from eval_anything.pipeline.t2t_task import T2TTask

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval_info', type=str, required=True, default='evaluate_test.yaml', help="YAML filename for evaluation configurations in eval-anything/configs/.")
    known_args, unparsed_args = parser.parse_known_args()
    unparsed_args = parse_unknown_args(unparsed_args)
    task = T2TTask(known_args.eval_info, **unparsed_args)
    task.iterate_run()
    
if __name__ == "__main__":
    main()

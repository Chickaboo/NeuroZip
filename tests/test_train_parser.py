import unittest

from neurozip.train import build_arg_parser


class TrainParserTests(unittest.TestCase):
    def test_accepts_torchrun_local_rank(self):
        args = build_arg_parser().parse_args(
            [
                "--train-path",
                "train.raw",
                "--valid-path",
                "valid.raw",
                "--output-dir",
                "artifacts",
                "--local-rank",
                "1",
            ]
        )
        self.assertEqual(args.local_rank, 1)


if __name__ == "__main__":
    unittest.main()

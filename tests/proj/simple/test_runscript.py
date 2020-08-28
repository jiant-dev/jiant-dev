import pytest
import argparse

from jiant.proj.simple import runscript as run
import jiant.scripts.download_data.runscript as downloader

@pytest.mark.parametrize("task_name", ["copa"])
@pytest.mark.parametrize("model_type", ["bert-base-cased"])
def test_simple_runscript(tmpdir, task_name, model_type):
	data_dir = str(tmpdir.mkdir('data'))
	exp_dir = str(tmpdir.mkdir('exp'))

	downloader.download_data([task_name], data_dir)

	cl_args = ["--run_name", task_name,
	           "--exp_dir", exp_dir,
	           "--data_dir", data_dir,
	           "--model_type", model_type,
	           "--tasks", task_name,
	           "--train_examples_cap", "32",
	           "--train_batch_size", "32",
	           "--no_cuda"]

	run_args = run.RunConfiguration.default_run_cli(cl_args=cl_args)
	run.run_simple(run_args)

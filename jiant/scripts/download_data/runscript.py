import os

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf
import jiant.scripts.download_data.datasets.glue as glue_download
import jiant.scripts.download_data.datasets.superglue as superglue_download
import jiant.scripts.download_data.datasets.xtreme as xtreme_download


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    output_base_path = zconf.attr(type=str)
    task_name_ls = zconf.attr(type=str, default=None)

    def _post_init(self):
        if isinstance(self.task_name_ls, str):
            self.task_name_ls = self.task_name_ls.split(",")


def download_data_and_write_config(
    task_name: str, task_data_base_path: str, task_config_base_path: str
):
    os.makedirs(task_data_base_path, exist_ok=True)
    os.makedirs(task_config_base_path, exist_ok=True)
    if task_name in [
        "cola",
        "sst",
        "mrpc",
        "qqp",
        "stsb",
        "mnli",
        "snli",
        "qnli",
        "rte",
        "wnli",
        "glue_diagnostics",
    ]:
        glue_download.download_glue_data_and_write_config(
            task_name=task_name,
            task_data_base_path=task_data_base_path,
            task_config_base_path=task_config_base_path,
        )
    elif task_name in [
        "cb",
        "copa",
        "multirc",
        "wic",
        "wsc",
        "boolq",
        "record",
        "superglue_broadcoverage_diagnostics",
        "superglue_winogender_diagnostics",
    ]:
        # Note: RTE handled by GLUE
        superglue_download.download_superglue_data_and_write_config(
            task_name=task_name,
            task_data_path=os.path.join(task_data_base_path, task_name),
            task_config_path=os.path.join(task_config_base_path, f"{task_name}_config.json"),
        )
    elif task_name in [
        "xnli",
        "pawsx",
        "udpos",
        "panx",
        "xquad",
        "mlqa",
        "tydiqa",
        "bucc2018",
        "tatoeba",
    ]:
        xtreme_download.download_xtreme_data_and_write_config(
            task_name=task_name,
            task_data_base_path=task_data_base_path,
            task_config_base_path=task_config_base_path,
        )
    else:
        raise KeyError(task_name)


def download_all_data(output_base_path: str, task_name_ls: list, verbose=True):
    for i, task_name in enumerate(task_name_ls):
        download_data_and_write_config(
            task_name=task_name,
            task_data_base_path=py_io.get_dir(output_base_path, "data"),
            task_config_base_path=py_io.get_dir(output_base_path, "configs"),
        )
        if verbose:
            print(f"Downloaded '{task_name}' ({i}/{len(task_name_ls)})")


def main():
    args = RunConfiguration.default_run_cli()
    download_all_data(
        output_base_path=args.output_base_path, task_name_ls=args.task_name_ls, verbose=True,
    )

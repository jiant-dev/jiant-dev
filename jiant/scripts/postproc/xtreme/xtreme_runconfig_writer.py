import jiant.proj.main.scripts.configurator as configurator
import jiant.utils.zconf as zconf
import jiant.utils.python.io as py_io
from jiant.utils.python.datastructures import get_unique_list_in_order


LANGS_DICT = {
    "xnli": "ar bg de el en es fr hi ru sw th tr ur vi zh".split(),
    "pawsx": "de en es fr ja ko zh".split(),
    "udpos": "af ar bg de el en es et eu fa fi fr he hi hu id it ja ko mr"
    " nl pt ru ta te tr ur vi zh".split(),
    "panx": "af ar bg bn de el en es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr ms my"
    " nl pt ru sw ta te th tl tr ur vi yo zh".split(),
    "xquad": "ar de el en es hi ru th tr vi zh".split(),
    "mlqa": "ar de en es hi vi zh".split(),
    "tydiqa": "ar bn en fi id ko ru sw te".split(),
    "bucc2018": "de fr ru zh".split(),
    "tatoeba": "af ar bg bn de el es et eu fa fi fr he hi hu id it ja jv ka kk ko ml mr"
    " nl pt ru sw ta te th tl tr ur vi zh".split(),
}
TRAIN_TASK_DICT = {
    "xnli": "mnli",
    "pawsx": "pawsx_en",
    "udpos": "udpos_en",
    "panx": "panx_en",
    "xquad": "squad_v1",
    "mlqa": "squad_v1",
    "tydiqa": "tydiqa_en",
}
TRAINED_TASKS = ["xnli", "pawsx", "udpos", "panx", "xquad", "mlqa", "tydiqa"]
UNTRAINED_TASKS = ["bucc2018", "tatoeba"]


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    xtreme_task = zconf.attr(type=str, required=True)
    task_config_base_path = zconf.attr(type=str, required=True)
    task_cache_base_path = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, required=True)

    # Optional
    epochs = zconf.attr(type=int, default=3)
    warmup_steps_proportion = zconf.attr(type=float, default=0.1)
    train_batch_size = zconf.attr(type=int, default=4)
    eval_batch_multiplier = zconf.attr(type=int, default=2)
    gradient_accumulation_steps = zconf.attr(type=int, default=1)
    eval_subset_num = zconf.attr(type=int, default=500)
    num_gpus = zconf.attr(type=int, default=1)
    early_stop_on_xtreme_tasks = zconf.attr(
        action="store_true",
        help="False = Do early stopping on train task,"
        " True = Do early stopping on XTREME tasks in all languages"
        " (default: False)",
    )
    no_verbose = zconf.attr(action="store_true")


def generate_configs(args: RunConfiguration):
    xtreme_task = args.xtreme_task
    if xtreme_task == "mlqa":
        xtreme_task_name_list = [f"{xtreme_task}_{lang}_{lang}" for lang in LANGS_DICT[xtreme_task]]
    else:
        xtreme_task_name_list = [f"{xtreme_task}_{lang}" for lang in LANGS_DICT[xtreme_task]]

    if xtreme_task in TRAINED_TASKS:
        train_task = TRAIN_TASK_DICT[xtreme_task]
        train_task_name_list = [train_task]
        val_task_name_list = get_unique_list_in_order([xtreme_task_name_list, train_task_name_list])
        if args.early_stop_on_xtreme_tasks:
            train_val_task_name_list = val_task_name_list
        else:
            train_val_task_name_list = train_task_name_list
    elif xtreme_task in UNTRAINED_TASKS:
        train_task_name_list = []
        val_task_name_list = xtreme_task_name_list
        train_val_task_name_list = []
    else:
        raise KeyError(xtreme_task)

    if not args.no_verbose:
        print("Training on:", ", ".join(train_task_name_list))
        print("Validation on:", ", ".join(val_task_name_list))
        print("Early stopping on:", ", ".join(train_val_task_name_list))

    config = configurator.SimpleAPIMultiTaskConfigurator(
        task_config_base_path=args.task_config_base_path,
        task_cache_base_path=args.task_cache_base_path,
        train_task_name_list=train_task_name_list,
        train_val_task_name_list=train_val_task_name_list,
        val_task_name_list=val_task_name_list,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_multiplier=args.eval_batch_multiplier,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_subset_num=args.eval_subset_num,
        num_gpus=args.num_gpus,
        warmup_steps_proportion=args.warmup_steps_proportion,
    ).create_config()

    # Make sure all tasks use the same task head
    config["taskmodels_config"]["task_to_taskmodel_map"] = {
        k: xtreme_task for k, v in config["taskmodels_config"]["task_to_taskmodel_map"].items()
    }
    if not args.no_verbose:
        print(f"Assigning all tasks to '{xtreme_task}' head")
    if xtreme_task in UNTRAINED_TASKS:
        # The reference implementation from the XTREME paper uses layer 14 for the
        #  retrieval representation.
        config["taskmodels_config"]["taskmodel_config_map"] = {
            xtreme_task: {"pooler_type": "mean", "layer": 14}
        }

    py_io.write_json(config, args.output_path)


if __name__ == "__main__":
    generate_configs(RunConfiguration.default_run_cli())

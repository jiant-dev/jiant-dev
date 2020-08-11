# Some datasets in nlp are vastly different than the original dataset. We opt
# to download the original dataset in this case.
SQUAD_TASKS = {"squad_v1", "squad_v2"}
INCOMPATIBLE_NLP_TASKS_TO_DATA_URLS = {
    "wsc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WSC.zip",
    "multirc": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/MultiRC.zip",
    "record": f"https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip",
}
INCOMPATIBLE_NLP_TASKS = INCOMPATIBLE_NLP_TASKS_TO_DATA_URLS.keys()

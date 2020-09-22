import transformers

import jiant.tasks as tasks


def test_tokenization():
    task = tasks.MLMPremaskedTask(name="mlm_premasked", path_dict={})
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    example = task.Example(
        guid=None,
        text="Hi, my name is Bob Roberts.",
        masked_spans=[[15, 18]],
    )
    tokenized_example = example.tokenize(tokenizer=tokenizer)
    assert tokenized_example.masked_tokens == \
        ['Hi', ',', 'Ġmy', 'Ġname', 'Ġis', 'Ġ', '<mask>', 'ĠRoberts', '.']
    assert tokenized_example.label_tokens == \
        ['<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', 'Bob', '<pad>', '<pad>']

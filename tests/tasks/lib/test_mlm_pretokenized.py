import transformers

import jiant.tasks as tasks


def test_tokenization():
    task = tasks.MLMPretokenizedTask(name="mlm_pretokenized", path_dict={})
    tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
    example = task.Example(
        guid=None,
        tokenized_text=['Hi', ',', 'Ġmy', 'Ġname', 'Ġis', 'ĠBob', 'ĠRoberts', '.'],
        masked_spans=[[2, 3], [5, 6]],
    )
    tokenized_example = example.tokenize(tokenizer=tokenizer)
    assert tokenized_example.masked_tokens == \
        ['Hi', ',', 'Ġmy', 'Ġname', 'Ġis', 'ĠBob', 'ĠRoberts', '.']
    assert tokenized_example.label_tokens == \
        ['<pad>', '<pad>', 'Ġmy', '<pad>', '<pad>', 'ĠBob', '<pad>', '<pad>']

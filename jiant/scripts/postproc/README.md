# NLP Benchmark Submission Formatter Scripts

## GLUE Benchmark Submission Formatter

Jiant supports [GLUE Benchmark](https://gluebenchmark.com/) test prediction generation in the submission format. To generate test predictions, use the `--write_test_preds` flag in [runscript.py](https://github.com/jiant-dev/jiant/blob/master/jiant/proj/main/runscript.py) when running your workflow. This will generate a `test_preds.p` file in the specified output directory. To convert `test_preds.p` to the required GLUE submission format, run the following:

```bash
python glue_submission_formatter.py --input_base_path $INPUT_BASE_PATH --output_path $OUTPUT_BASE PATH
```

where `$INPUT_BASE_PATH` contains the task folder(s) output by [runscript.py](https://github.com/jiant-dev/jiant/blob/master/jiant/proj/main/runscript.py). Alternatively, a subset of tasks can be formatted using:

```bash
python glue_submission_formatter.py --tasks cola mrpc --input_base_path $INPUT_BASE_PATH --output_path $OUTPUT_BASE PATH
```

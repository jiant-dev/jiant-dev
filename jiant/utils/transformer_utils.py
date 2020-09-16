import contextlib

from jiant.shared.model_resolution import ModelArchitectures


@contextlib.contextmanager
def output_hidden_states_context(encoder):
    model_arch = ModelArchitectures.from_encoder(encoder)
    if model_arch in (ModelArchitectures.BART, ModelArchitectures.MBART):
        yield
        return
    elif model_arch in (
        ModelArchitectures.BERT,
        ModelArchitectures.ROBERTA,
        ModelArchitectures.ALBERT,
        ModelArchitectures.XLM_ROBERTA,
    ):
        if hasattr(encoder.encoder, "output_hidden_states"):
            # Transformers < v2
            modified_obj = encoder.encoder
        elif hasattr(encoder.encoder.config, "output_hidden_states"):
            # Transformers >= v3
            modified_obj = encoder.encoder.config
        else:
            raise RuntimeError(f"Failed to convert model {type(encoder)} to output hidden states")
    else:
        raise KeyError(model_arch)

    old_value = modified_obj.output_hidden_states
    modified_obj.output_hidden_states = True
    yield
    modified_obj.output_hidden_states = old_value

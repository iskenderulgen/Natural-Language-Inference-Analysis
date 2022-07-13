"""
Since 3.1.0 version old init-model approach is moved to init_pipeline.py file.
This code enables static word vectors to be implemented into spacy NLP object.
Thus, the NLP analyst can maintain project from one framework by loading embeddings to NLP object.
For easy manipulation. Args() parameters are re-designed with argparse. Thus enables to use everything
from top of the script.
"""
import argparse
from typing import Optional
import logging
from pathlib import Path
from wasabi import msg
import typer
import srsly

from spacy import util
from spacy.training.initialize import init_nlp, convert_vectors
from spacy.language import Language
from spacy.cli._util import init_cli, Arg, Opt, parse_config_overrides, show_validation_error
from spacy.cli._util import import_code, setup_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="en",
                    help="Keys language of the pretrained vectors, The language of the nlp object to create")
parser.add_argument("--vectors_location", type=str,
                    default="/media/ulgen/Samsung/contradiction_data_depo/zips/googlenews_txt_format.zip",
                    help="Path for the pretrained vectors to be imported")
parser.add_argument("--output_dir", type=str, default="/media/ulgen/Samsung/contradiction_data/transformers/word2vec/",
                    help="Path for the imported NLP object")
parser.add_argument("--prune", type=int, default=1500000,
                    help="Number of vectors to be pruned. This is an exhaustive approach that assigns semantically"
                         "similarly keys to same vector")
parser.add_argument("--name", type=str, default="Word2Vec",
                    help="Name for the imported vectors. Default use vector source name")
parser.add_argument("--verbose", type=bool, default=True,
                    help="Verbosity of the process")
args = parser.parse_args()


def init_vectors_cli(
        lang: str = args.language,
        vectors_loc: Path = args.vectors_location,
        output_dir: Path = args.output_dir,
        prune: int = args.prune,
        truncate: int = 0,
        name: Optional[str] = args.name,
        verbose: bool = args.verbose,
        jsonl_loc: Optional[Path] = None):
    """Convert word vectors for use with spaCy. Will export an nlp object that
    you can use in the [initialize] block of your config to initialize
    a model with vectors.
    """
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    msg.info(f"Creating blank nlp object for language '{lang}'")
    nlp = util.get_lang_class(lang)()
    if jsonl_loc is not None:
        update_lexemes(nlp, jsonl_loc)
    convert_vectors(nlp, vectors_loc, truncate=truncate, prune=prune, name=name)
    msg.good(f"Successfully converted {len(nlp.vocab.vectors)} vectors")
    nlp.to_disk(output_dir)


def update_lexemes(nlp: Language, jsonl_loc: Path) -> None:
    # Mostly used for backwards-compatibility and may be removed in the future
    lex_attrs = srsly.read_jsonl(jsonl_loc)
    for attrs in lex_attrs:
        if "settings" in attrs:
            continue
        lexeme = nlp.vocab[attrs["orth"]]
        lexeme.set_attrs(**attrs)


@init_cli.command(
    "nlp",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    hidden=True,
)
def init_pipeline_cli(
        # fmt: off
        ctx: typer.Context,  # This is only used to read additional arguments
        config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
        output_path: Path = Arg(..., help="Output directory for the prepared data"),
        code_path: Optional[Path] = Opt(None, "--code", "-c",
                                        help="Path to Python file with additional code (registered functions) to be imported"),
        verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
        use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
        # fmt: on
):
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    setup_gpu(use_gpu)
    with show_validation_error(config_path):
        config = util.load_config(config_path, overrides=overrides)
    with show_validation_error(hint_fill=False):
        nlp = init_nlp(config, use_gpu=use_gpu)
    nlp.to_disk(output_path)
    msg.good(f"Saved initialized pipeline to {output_path}")


@init_cli.command(
    "labels",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def init_labels_cli(
        # fmt: off
        ctx: typer.Context,  # This is only used to read additional arguments
        config_path: Path = Arg(..., help="Path to config file", exists=True, allow_dash=True),
        output_path: Path = Arg(..., help="Output directory for the labels"),
        code_path: Optional[Path] = Opt(None, "--code", "-c",
                                        help="Path to Python file with additional code (registered functions) to be imported"),
        verbose: bool = Opt(False, "--verbose", "-V", "-VV", help="Display more information for debugging purposes"),
        use_gpu: int = Opt(-1, "--gpu-id", "-g", help="GPU ID or -1 for CPU")
        # fmt: on
):
    """Generate JSON files for the labels in the data. This helps speed up the
    training process, since spaCy won't have to preprocess the data to
    extract the labels."""
    util.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    overrides = parse_config_overrides(ctx.args)
    import_code(code_path)
    setup_gpu(use_gpu)
    with show_validation_error(config_path):
        config = util.load_config(config_path, overrides=overrides)
    with show_validation_error(hint_fill=False):
        nlp = init_nlp(config, use_gpu=use_gpu)
    _init_labels(nlp, output_path)


def _init_labels(nlp, output_path):
    for name, component in nlp.pipeline:
        if getattr(component, "label_data", None) is not None:
            output_file = output_path / f"{name}.json"
            srsly.write_json(output_file, component.label_data)
            msg.good(f"Saving label data for component '{name}' to {output_file}")
        else:
            msg.info(f"No label data found for component '{name}'")


init_vectors_cli()

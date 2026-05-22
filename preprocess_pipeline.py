#!/usr/bin/env python3
"""Unified preprocessing entrypoint for DeepSecReview."""

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

import config
from config import ConfDatasets
from preprocess import (
    extract_abstract,
    extract_abstract_pipeline,
    generate_normalize_list,
    pdf2txt_parellel,
    run_normalize_parallel,
)


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = "pipeline.log"
FLAG_DEBUG = False

logger = logging.getLogger(__name__)


def configure_logging(log_file: str = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configure root logging so all imported modules share the same handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def count_text_files(directory: str | Path) -> int:
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob("*.txt")))


def generate_missing_normalize_tasks(dataset_list, doc_type: str):
    tasks = generate_normalize_list(dataset_list, doc_type=doc_type)
    return [task for task in tasks if not Path(task[1]).exists()]


def run_conference_preprocessing(debug: bool = FLAG_DEBUG, n_thread: int = 10):
    """Run the original conference preprocessing pipeline."""
    datasets_to_process = ConfDatasets

    logger.info("-" * 80)
    logger.info("[1/4] Conference PDF to TXT")
    logger.info("-" * 80)
    for dataset in datasets_to_process:
        logger.info("Processing dataset: %s", dataset.name)
        pdf2txt_parellel(dataset=dataset, n_thread=n_thread, DEBUG=debug)

    logger.info("-" * 80)
    logger.info("[2/4] Conference abstract extraction")
    logger.info("-" * 80)
    total_papers = sum(len(ds) for ds in datasets_to_process)
    with tqdm(total=total_papers, desc="Extract abstracts") as tqdm_bar:
        counter = 0
        for dataset in datasets_to_process:
            counter = extract_abstract_pipeline(
                dataset,
                tqdm_bar=tqdm_bar,
                counter=counter,
                DEBUG=debug,
            )
            if debug:
                break

    logger.info("-" * 80)
    logger.info("[3/4] Conference full-text normalization")
    logger.info("-" * 80)
    txt_tasks = generate_normalize_list(datasets_to_process, doc_type="txt")
    run_normalize_parallel(txt_tasks)

    logger.info("-" * 80)
    logger.info("[4/4] Conference abstract normalization")
    logger.info("-" * 80)
    abs_tasks = generate_normalize_list(datasets_to_process, doc_type="abs")
    run_normalize_parallel(abs_tasks)


def build_missing_arxiv_txt_pp_tasks(dataset):
    tasks = []
    for paper_id in dataset.filter_list:
        slug = config.ArxivDatasetConfig.slugify(paper_id, "txt")
        txt_file = Path(dataset.txt_dir) / slug
        txt_pp_file = Path(dataset.txt_pp_dir) / slug
        if txt_file.exists() and not txt_pp_file.exists():
            tasks.append((str(txt_file), str(txt_pp_file), dataset.name))
    return tasks


def build_missing_arxiv_abs_pairs(dataset):
    pairs = []
    for paper_id in dataset.filter_list:
        slug = config.ArxivDatasetConfig.slugify(paper_id, "txt")
        txt_file = Path(dataset.txt_dir) / slug
        abs_file = Path(dataset.abs_dir) / slug
        if txt_file.exists() and not abs_file.exists():
            pairs.append((txt_file, abs_file))
    return pairs


def build_missing_arxiv_abs_pp_tasks(dataset):
    tasks = []
    for paper_id in dataset.filter_list:
        slug = config.ArxivDatasetConfig.slugify(paper_id, "txt")
        abs_file = Path(dataset.abs_dir) / slug
        abs_pp_file = Path(dataset.abs_pp_dir) / slug
        if abs_file.exists() and not abs_pp_file.exists():
            tasks.append((str(abs_file), str(abs_pp_file), dataset.name))
    return tasks


def extract_missing_arxiv_abstracts(dataset, debug: bool = FLAG_DEBUG):
    missing_pairs = build_missing_arxiv_abs_pairs(dataset)
    if not missing_pairs:
        logger.info("No missing arXiv abstracts found.")
        return 0

    logger.info("Extracting %s missing arXiv abstracts.", len(missing_pairs))
    for txt_file, abs_file in tqdm(missing_pairs, desc="Extract arXiv abstracts"):
        with open(txt_file, "r", encoding="utf-8", errors="ignore") as src:
            abstract = extract_abstract(src.read())
        if debug:
            logger.info("Debug abstract preview for %s: %s", txt_file.name, abstract[:200])
            break
        with open(abs_file, "w", encoding="utf-8") as dst:
            dst.write(abstract)
    return len(missing_pairs)


def repair_arxiv_negative_samples(
    repair_txt_pp: bool = True,
    repair_abs: bool = True,
    repair_abs_pp: bool = True,
    debug: bool = FLAG_DEBUG,
):
    """Repair preprocessing artifacts for arXiv negative samples."""
    arxiv_dataset = config.ArxivDataset

    logger.info("=" * 80)
    logger.info("Repairing arXiv negative samples")
    logger.info("=" * 80)
    logger.info("Dataset: %s", arxiv_dataset.name)
    logger.info("Root: %s", arxiv_dataset.root_dir)

    txt_count = count_text_files(arxiv_dataset.txt_dir)
    txt_pp_count = count_text_files(arxiv_dataset.txt_pp_dir)
    abs_count = count_text_files(arxiv_dataset.abs_dir)
    abs_pp_count = count_text_files(arxiv_dataset.abs_pp_dir)

    logger.info(
        "Current status: txt=%s, txt_pp=%s, abs=%s, abs_pp=%s",
        txt_count,
        txt_pp_count,
        abs_count,
        abs_pp_count,
    )

    missing_txt = []
    for paper_id in arxiv_dataset.filter_list:
        slug = config.ArxivDatasetConfig.slugify(paper_id, "txt")
        txt_file = Path(arxiv_dataset.txt_dir) / slug
        if not txt_file.exists():
            missing_txt.append(paper_id)

    if missing_txt:
        logger.warning("Missing %s arXiv TXT files.", len(missing_txt))
        for paper_id in missing_txt[:10]:
            logger.warning("Missing TXT: %s", paper_id)

    if repair_txt_pp:
        txt_tasks = build_missing_arxiv_txt_pp_tasks(arxiv_dataset)
        if txt_tasks:
            logger.info("Normalizing %s missing arXiv full texts.", len(txt_tasks))
            run_normalize_parallel(txt_tasks)
        else:
            logger.info("No missing arXiv full-text normalization tasks found.")

    if repair_abs:
        extract_missing_arxiv_abstracts(arxiv_dataset, debug=debug)

    if repair_abs_pp:
        abs_tasks = build_missing_arxiv_abs_pp_tasks(arxiv_dataset)
        if abs_tasks:
            logger.info("Normalizing %s missing arXiv abstracts.", len(abs_tasks))
            run_normalize_parallel(abs_tasks)
        else:
            logger.info("No missing arXiv abstract normalization tasks found.")

    final_status = {
        "txt": count_text_files(arxiv_dataset.txt_dir),
        "txt_pp": count_text_files(arxiv_dataset.txt_pp_dir),
        "abs": count_text_files(arxiv_dataset.abs_dir),
        "abs_pp": count_text_files(arxiv_dataset.abs_pp_dir),
        "missing_txt": len(missing_txt),
    }
    logger.info("Final arXiv status: %s", final_status)
    return final_status


def build_arg_parser():
    parser = argparse.ArgumentParser(description="DeepSecReview unified preprocessing pipeline")
    parser.add_argument(
        "--mode",
        choices=("full", "conference", "arxiv", "arxiv_txt", "arxiv_abs"),
        default="full",
        help=(
            "full: conference preprocessing + all arXiv repairs; "
            "conference: conference preprocessing only; "
            "arxiv: all arXiv repairs only; "
            "arxiv_txt: repair arXiv full-text normalization only; "
            "arxiv_abs: repair arXiv abstract extraction and abstract normalization only."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help="Log file path. Defaults to pipeline.log.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    configure_logging(args.log_file)

    logger.info("=" * 80)
    logger.info("Starting DeepSecReview preprocessing pipeline")
    logger.info("Mode: %s", args.mode)
    logger.info("=" * 80)

    if args.mode in ("full", "conference"):
        run_conference_preprocessing(debug=FLAG_DEBUG)

    if args.mode == "full":
        repair_arxiv_negative_samples(debug=FLAG_DEBUG)
    elif args.mode == "arxiv":
        repair_arxiv_negative_samples(debug=FLAG_DEBUG)
    elif args.mode == "arxiv_txt":
        repair_arxiv_negative_samples(
            repair_txt_pp=True,
            repair_abs=False,
            repair_abs_pp=False,
            debug=FLAG_DEBUG,
        )
    elif args.mode == "arxiv_abs":
        repair_arxiv_negative_samples(
            repair_txt_pp=False,
            repair_abs=True,
            repair_abs_pp=True,
            debug=FLAG_DEBUG,
        )

    logger.info("=" * 80)
    logger.info("All preprocessing stages completed.")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

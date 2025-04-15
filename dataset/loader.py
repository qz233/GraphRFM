from .arxiv import get_arxiv_dataset


def get_dataset(config):
    if config.dataset in ["arxiv", "arxiv-year"]:
        return get_arxiv_dataset(config.dataset, config.dataset_path)
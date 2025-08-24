import logging

def setup_logging(level:str="INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s"
    )
    return logging.getLogger("recmatcher")

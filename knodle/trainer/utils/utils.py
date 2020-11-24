import logging


def print_section(text: str, logger: logging) -> None:
    """
    Prints a section
    Args:
        text: Text to print
        logger: Logger object

    Returns:

    """
    logger.info("======================================")
    logger.info(text)
    logger.info("======================================")

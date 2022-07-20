import logging


def silence_smac_loggers():
    logger_names = list(logging.root.manager.loggerDict)
    logger_names = [n for n in logger_names if n.startswith("smac")]

    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level=logging.ERROR)
        logger.propagate = False

        # _handlers = logger.handlers
        # for _handler in _handlers:
        #     logger.removeHandler(_handler)

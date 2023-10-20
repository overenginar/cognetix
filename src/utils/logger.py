
class Logger:

    def __init__(self, sc, logger_name):
        log4jLogger = sc._jvm.org.apache.log4j
        self.logger = log4jLogger.LogManager.getLogger(logger_name)

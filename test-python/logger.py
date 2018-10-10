import logging

# create logger
logger = logging.getLogger('simple_example')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

fl = logging.FileHandler('logger.log')
fl.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s \n%(filename)s - %(module)s - %(funcName)s - %(lineno)d \n%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

# add formatter to ch
ch.setFormatter(formatter)
fl.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fl)

# 'application' code
logger.debug('debug message')
logger.info('info message')
logger.warn('warn message')
logger.error('error message')
logger.critical('critical message')
logger.info('test')

logger.setLevel(logging.INFO)

b = 100

logger.info('b is {}'.format(b))

logger.warning('warning !')

def func():
    logger.debug('debug in func')
    logger.info('in this function')
    logger.warning('in this function!!!!')
    logger.debug(logger.findCaller())

func()
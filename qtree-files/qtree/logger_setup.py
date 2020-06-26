import logging

log = logging.getLogger('qtree')
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    '%(asctime)s- %(levelname)s•\t%(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

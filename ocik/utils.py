##################
# PORTE LOGIQUE  #
#################


def f_and(*args):
    return int(all(args))


def f_or(*args):
    return int(any(args))


def f_not(a):
    return int(not a)


def f_nor(a, b):
    return int(not (f_or(a, b)))


def f_nand(a, b):
    return int(not (f_and(a, b)))


def f_xor(a, b):
    return int(a ^ b)

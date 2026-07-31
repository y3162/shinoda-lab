GREEN = '\033[1;32m'
YELLOW = '\033[1;33m'
RED = '\033[1;31m'
RESET = '\033[0m'


def print_log(
    *args,
    **kwargs,
):
    print(f'{GREEN}[LOG]{RESET}', end=' ')
    print(*args, **kwargs)


def print_warning(
    *args,
    **kwargs,
):
    print(f'{YELLOW}[WARNING]{RESET}', end=' ')
    print(*args, **kwargs)


def print_error(
    *args,
    **kwargs,
):
    print(f'{RED}[ERROR]{RESET}', end=' ')
    print(*args, **kwargs)

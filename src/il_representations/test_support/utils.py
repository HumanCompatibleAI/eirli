import io

from il_representations import algos


def is_representation_learner(el):
    try:
        return issubclass(el, algos.RepresentationLearner) and el != algos.RepresentationLearner and el not in algos.WIP_ALGOS
    except TypeError:
        return False


def files_are_identical(p1, p2, read_size=(1 << 16) - 1):
    """Figure out whether paths p1 and p2 point to an identical file.
    Takes the dumb approach of reading the entire file out, but this make it
    robust to choice of whether file is copied vs. symlinked vs. hardlinked.

    Note that this only works for ordinary files, and not sockets etc.; for
    'interactive' files, `BufferedReader.read()` might return a short read
    before EOF, in which case this function will break."""
    with open(p1, 'rb') as fp1, open(p2, 'rb') as fp2:
        r1 = io.BufferedReader(fp1, read_size)
        r2 = io.BufferedReader(fp2, read_size)
        while True:
            fp1_contents = r1.read(read_size)
            fp2_contents = r2.read(read_size)
            if fp1_contents != fp2_contents:
                return False
            if not fp1_contents or not fp2_contents:
                # EOF
                assert not fp1_contents and not fp2_contents
                break
    return True

# -*- coding: utf-8 -*-

#
# IceCream - Never use print() to debug again
#
# Ansgar Grunseid
# grunseid.com
# grunseid@gmail.com
#
# License: MIT
#

import sys
import unittest
import warnings

from io import StringIO
from contextlib import contextmanager
from os.path import basename, splitext, realpath

import icecream
from icecream import ic, argumentToString, stderrPrint, NO_SOURCE_AVAILABLE_WARNING_MESSAGE
from icecream.icecream import is_sensitive_key, _mask_sensitive_value

TEST_PAIR_DELIMITER = '| '
MY_FILENAME = basename(__file__)
MY_FILEPATH = realpath(__file__)


a = 1
b = 2
c = 3


def noop(*args, **kwargs):
    return


def has_ansi_escape_codes(s):
    # Oversimplified, but ¯\_(ツ)_/¯. TODO(grun): Test with regex.
    return '\x1b[' in s


class FakeTeletypeBuffer(StringIO):
    """
    Extend StringIO to act like a TTY so ANSI control codes aren't stripped
    when wrapped with colorama's wrap_stream().
    """
    def isatty(self):
        return True


@contextmanager
def disable_coloring():
    originalOutputFunction = ic.outputFunction

    ic.configureOutput(outputFunction=stderrPrint)
    yield
    ic.configureOutput(outputFunction=originalOutputFunction)


@contextmanager
def configure_icecream_output(prefix=None, outputFunction=None,
                            argToStringFunction=None, includeContext=None,
                            contextAbsPath=None):
    oldPrefix = ic.prefix
    oldOutputFunction = ic.outputFunction
    oldArgToStringFunction = ic.argToStringFunction
    oldIncludeContext = ic.includeContext
    oldContextAbsPath = ic.contextAbsPath

    if prefix:
        ic.configureOutput(prefix=prefix)
    if outputFunction:
        ic.configureOutput(outputFunction=outputFunction)
    if argToStringFunction:
        ic.configureOutput(argToStringFunction=argToStringFunction)
    if includeContext:
        ic.configureOutput(includeContext=includeContext)
    if contextAbsPath:
        ic.configureOutput(contextAbsPath=contextAbsPath)

    yield

    ic.configureOutput(
        oldPrefix, oldOutputFunction, oldArgToStringFunction,
        oldIncludeContext, oldContextAbsPath)


@contextmanager
def capture_standard_streams():
    realStdout = sys.stdout
    realStderr = sys.stderr
    newStdout = FakeTeletypeBuffer()
    newStderr = FakeTeletypeBuffer()
    try:
        sys.stdout = newStdout
        sys.stderr = newStderr
        yield newStdout, newStderr
    finally:
        sys.stdout = realStdout
        sys.stderr = realStderr


def strip_prefix(line):
    if line.startswith(ic.prefix):
        line = line.strip()[len(ic.prefix):]
    return line


def line_is_context_and_time(line):
    line = strip_prefix(line)  # ic| f.py:33 in foo() at 08:08:51.389
    context, time = line.split(' at ')

    return (
        line_is_context(context) and
        len(time.split(':')) == 3 and
        len(time.split('.')) == 2)


def line_is_context(line):
    line = strip_prefix(line)  # ic| f.py:33 in foo()
    sourceLocation, function = line.split(' in ')  # f.py:33 in foo()
    filename, lineNumber = sourceLocation.split(':')  # f.py:33
    name, ext = splitext(filename)

    return (
        int(lineNumber) > 0 and
        ext in ['.py', '.pyc', '.pyo'] and
        name == splitext(MY_FILENAME)[0] and
        (function == '<module>' or function.endswith('()')))

def line_is_abs_path_context(line):
    line = strip_prefix(line)  # ic| /absolute/path/to/f.py:33 in foo()
    sourceLocation, function = line.split(' in ')  # /absolute/path/to/f.py:33 in foo()
    filepath, lineNumber = sourceLocation.split(':')  # /absolute/path/to/f.py:33
    path, ext = splitext(filepath)

    return (
        int(lineNumber) > 0 and
        ext in ['.py', '.pyc', '.pyo'] and
        path == splitext(MY_FILEPATH)[0] and
        (function == '<module>' or function.endswith('()')))

def line_after_context(line, prefix):
    if line.startswith(prefix):
        line = line[len(prefix):]

    toks = line.split(' in ', 1)
    if len(toks) == 2:
        rest = toks[1].split(' ')
        line = ' '.join(rest[1:])

    return line

def parse_output_into_pairs(out, err, assert_num_lines,
                            prefix=icecream.DEFAULT_PREFIX):
    if isinstance(out, StringIO):
        out = out.getvalue()
    if isinstance(err, StringIO):
        err = err.getvalue()

    assert not out

    lines = err.splitlines()
    if assert_num_lines:
        assert len(lines) == assert_num_lines

    line_pairs = []
    for line in lines:
        line = line_after_context(line, prefix)

        if not line:
            line_pairs.append([])
            continue

        pairStrs = line.split(TEST_PAIR_DELIMITER)
        pairs = [tuple(s.split(':', 1)) for s in pairStrs]
        # Indented line of a multiline value.
        if len(pairs[0]) == 1 and line.startswith(' '):
            arg, value = line_pairs[-1][-1]
            looksLikeAString = value[0] in ["'", '"']
            prefix = ((arg + ': ' if arg is not None else '')  # A multiline value
                      + (' ' if looksLikeAString else ''))
            dedented = line[len(ic.prefix) + len(prefix):]
            line_pairs[-1][-1] = (arg, value + '\n' + dedented)
        else:
            items = [
                (None, p[0].strip()) if len(p) == 1  # A value, like ic(3).
                else (p[0].strip(), p[1].strip())  # A variable, like ic(a).
                for p in pairs]
            line_pairs.append(items)

    return line_pairs


class TestIceCream(unittest.TestCase):
    def setUp(self):
        ic._pairDelimiter = TEST_PAIR_DELIMITER

    def test_metadata(self):
        def is_non_empty_string(s):
            return isinstance(s, str) and s
        assert is_non_empty_string(icecream.__title__)
        assert is_non_empty_string(icecream.__version__)
        assert is_non_empty_string(icecream.__license__)
        assert is_non_empty_string(icecream.__author__)
        assert is_non_empty_string(icecream.__contact__)
        assert is_non_empty_string(icecream.__description__)
        assert is_non_empty_string(icecream.__url__)

    def test_without_args(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic()
        assert line_is_context_and_time(err.getvalue())

    def test_as_argument(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            noop(ic(a), ic(b))
        pairs = parse_output_into_pairs(out, err, 2)
        assert pairs[0][0] == ('a', '1') and pairs[1][0] == ('b', '2')

        with disable_coloring(), capture_standard_streams() as (out, err):
            dic = {1: ic(a)}  # noqa
            lst = [ic(b), ic()]  # noqa
        pairs = parse_output_into_pairs(out, err, 3)
        assert pairs[0][0] == ('a', '1')
        assert pairs[1][0] == ('b', '2')
        assert line_is_context_and_time(err.getvalue().splitlines()[-1])

    def test_single_argument(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)
        assert parse_output_into_pairs(out, err, 1)[0][0] == ('a', '1')

    def test_multiple_arguments(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a, b)
        pairs = parse_output_into_pairs(out, err, 1)[0]
        assert pairs == [('a', '1'), ('b', '2')]

    def test_nested_multiline(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(
                )
        assert line_is_context_and_time(err.getvalue())

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a,
               'foo')
        pairs = parse_output_into_pairs(out, err, 1)[0]
        assert pairs == [('a',  '1'), (None, "'foo'")]

        with disable_coloring(), capture_standard_streams() as (out, err):
            noop(noop(noop({1: ic(
                noop())})))
        assert parse_output_into_pairs(out, err, 1)[0][0] == ('noop()', 'None')

    def test_expression_arguments(self):
        class klass():
            attr = 'yep'
        d = {'d': {1: 'one'}, 'k': klass}

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(d['d'][1])
        pair = parse_output_into_pairs(out, err, 1)[0][0]
        assert pair == ("d['d'][1]", "'one'")

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(d['k'].attr)
        pair = parse_output_into_pairs(out, err, 1)[0][0]
        assert pair == ("d['k'].attr", "'yep'")

    def test_multiple_calls_on_same_line(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a); ic(b, c)  # noqa
        pairs = parse_output_into_pairs(out, err, 2)
        assert pairs[0][0] == ('a', '1')
        assert pairs[1] == [('b', '2'), ('c', '3')]

    def test_call_surrounded_by_expressions(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            noop(); ic(a); noop()  # noqa
        assert parse_output_into_pairs(out, err, 1)[0][0] == ('a', '1')

    def test_comments(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            """Comment."""; ic(); # Comment.  # noqa
        assert line_is_context_and_time(err.getvalue())

    def test_method_arguments(self):
        class Foo:
            def foo(self):
                return 'foo'
        f = Foo()
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(f.foo())
        assert parse_output_into_pairs(out, err, 1)[0][0] == ('f.foo()', "'foo'")

    def test_complicated(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            noop(); ic(); noop(); ic(a,  # noqa
                                     b, noop.__class__.__name__,  # noqa
                                         noop ()); noop()  # noqa
        pairs = parse_output_into_pairs(out, err, 2)
        assert line_is_context_and_time(err.getvalue().splitlines()[0])
        assert pairs[1] == [
            ('a', '1'), ('b', '2'), ('noop.__class__.__name__', "'function'"),
            ('noop ()', 'None')]

    def test_return_value(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            assert ic() is None
            assert ic(1) == 1
            assert ic(1, 2, 3) == (1, 2, 3)

    def test_different_name(self):
        from icecream import ic as foo
        with disable_coloring(), capture_standard_streams() as (out, err):
            foo()
        assert line_is_context_and_time(err.getvalue())

        newname = foo
        with disable_coloring(), capture_standard_streams() as (out, err):
            newname(a)
        pair = parse_output_into_pairs(out, err, 1)[0][0]
        assert pair == ('a', '1')

    def test_prefix_configuration(self):
        prefix = 'lolsup '
        with configure_icecream_output(prefix, stderrPrint):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
        pair = parse_output_into_pairs(out, err, 1, prefix=prefix)[0][0]
        assert pair == ('a', '1')

        def prefix_function():
            return 'lolsup '
        with configure_icecream_output(prefix=prefix_function):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
        pair = parse_output_into_pairs(out, err, 1, prefix=prefix_function())[0][0]
        assert pair == ('b', '2')

    def test_output_function(self):
        lst = []

        def append_to(s):
            lst.append(s)

        with configure_icecream_output(ic.prefix, append_to):
            with capture_standard_streams() as (out, err):
                ic(a)
        assert not out.getvalue() and not err.getvalue()

        with configure_icecream_output(outputFunction=append_to):
            with capture_standard_streams() as (out, err):
                ic(b)
        assert not out.getvalue() and not err.getvalue()

        pairs = parse_output_into_pairs(out, '\n'.join(lst), 2)
        assert pairs == [[('a', '1')], [('b', '2')]]

    def test_enable_disable(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            assert ic(a) == 1
            assert ic.enabled

            ic.disable()
            assert not ic.enabled
            assert ic(b) == 2

            ic.enable()
            assert ic.enabled
            assert ic(c) == 3

        pairs = parse_output_into_pairs(out, err, 2)
        assert pairs == [[('a', '1')], [('c', '3')]]

    def test_arg_to_string_function(self):
        def hello(obj):
            return 'zwei'

        with configure_icecream_output(argToStringFunction=hello):
            with disable_coloring(), capture_standard_streams() as (out, err):
                eins = 'ein'
                ic(eins)
        pair = parse_output_into_pairs(out, err, 1)[0][0]
        assert pair == ('eins', 'zwei')

    def test_singledispatch_argument_to_string(self):
        def argument_to_string_tuple(obj):
            return "Dispatching tuple!"

        # Prepare input and output
        x = (1, 2)
        default_output = ic.format(x)

        # Register
        argumentToString.register(tuple, argument_to_string_tuple)
        assert tuple in argumentToString.registry
        assert str.endswith(ic.format(x), argument_to_string_tuple(x))

        # Unregister
        argumentToString.unregister(tuple)
        assert tuple not in argumentToString.registry
        assert ic.format(x) == default_output

    def test_single_argument_long_line_not_wrapped(self):
        # A single long line with one argument is not line wrapped.
        longStr = '*' * (ic.lineWrapWidth + 1)
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(longStr)
        pair = parse_output_into_pairs(out, err, 1)[0][0]
        assert len(err.getvalue()) > ic.lineWrapWidth
        assert pair == ('longStr', ic.argToStringFunction(longStr))

    def test_multiple_arguments_long_line_wrapped(self):
        # A single long line with multiple variables is line wrapped.
        val = '*' * int(ic.lineWrapWidth / 4)
        valStr = ic.argToStringFunction(val)

        v1 = v2 = v3 = v4 = val
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(v1, v2, v3, v4)

        pairs = parse_output_into_pairs(out, err, 4)
        assert pairs == [[(k, valStr)] for k in ['v1', 'v2', 'v3', 'v4']]

        lines = err.getvalue().splitlines()
        assert (
            lines[0].startswith(ic.prefix) and
            lines[1].startswith(' ' * len(ic.prefix)) and
            lines[2].startswith(' ' * len(ic.prefix)) and
            lines[3].startswith(' ' * len(ic.prefix)))

    def test_multiline_value_wrapped(self):
        # Multiline values are line wrapped.
        multilineStr = 'line1\nline2'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(multilineStr)
        pair = parse_output_into_pairs(out, err, 2)[0][0]
        assert pair == ('multilineStr', ic.argToStringFunction(multilineStr))

    def test_include_context_single_line(self):
        i = 3
        with configure_icecream_output(includeContext=True):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(i)

        pair = parse_output_into_pairs(out, err, 1)[0][0]
        assert pair == ('i', '3')

    def test_context_abs_path_single_line(self):
        i = 3
        with configure_icecream_output(includeContext=True, contextAbsPath=True):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(i)
        # Output with absolute path can easily exceed line width, so no assert line num here.
        pairs = parse_output_into_pairs(out, err, 0)
        assert [('i', '3')] in pairs

    def test_values(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            # Test both 'asdf' and "asdf"; see
            # https://github.com/gruns/icecream/issues/53.
            ic(3, 'asdf', "asdf")

        pairs = parse_output_into_pairs(out, err, 1)
        assert pairs == [[(None, '3'), (None, "'asdf'"), (None, "'asdf'")]]

    def test_include_context_multi_line(self):
        multilineStr = 'line1\nline2'
        with configure_icecream_output(includeContext=True):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(multilineStr)

        firstLine = err.getvalue().splitlines()[0]
        assert line_is_context(firstLine)

        pair = parse_output_into_pairs(out, err, 3)[1][0]
        assert pair == ('multilineStr', ic.argToStringFunction(multilineStr))

    def test_context_abs_path_multi_line(self):
        multilineStr = 'line1\nline2'
        with configure_icecream_output(includeContext=True, contextAbsPath=True):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(multilineStr)

        firstLine = err.getvalue().splitlines()[0]
        assert line_is_abs_path_context(firstLine)

        pair = parse_output_into_pairs(out, err, 3)[1][0]
        assert pair == ('multilineStr', ic.argToStringFunction(multilineStr))

    def test_format(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            """comment"""; noop(); ic(  # noqa
                'sup'); noop()  # noqa
        """comment"""; noop(); s = ic.format(  # noqa
            'sup'); noop()  # noqa
        assert s == err.getvalue().rstrip()

    def test_multiline_invocation_with_comments(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(  # Comment.

                a,  # Comment.

                # Comment.

                b,  # Comment.

                )  # Comment.

        pairs = parse_output_into_pairs(out, err, 1)[0]
        assert pairs == [('a', '1'), ('b', '2')]

    def test_no_source_available_prints_values(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            with warnings.catch_warnings():
                # we ignore the warning so that it doesn't interfere
                # with parsing ic's output
                warnings.simplefilter("ignore")
                eval('ic(a, b)')
                pairs = parse_output_into_pairs(out, err, 1)
                self.assertEqual(pairs, [[(None, '1'), (None, "2")]])

    def test_no_source_available_prints_multiline(self):
        """
        This tests for a bug which caused only multiline prints to fail.
        """
        multilineStr = 'line1\nline2'
        with disable_coloring(), capture_standard_streams() as (out, err):
            with warnings.catch_warnings():
                # we ignore the warning so that it doesn't interfere
                # with parsing ic's output
                warnings.simplefilter("ignore")
                eval('ic(multilineStr)')
                pair = parse_output_into_pairs(out, err, 2)[0][0]
                self.assertEqual(pair, (None, ic.argToStringFunction(multilineStr)))

    def test_no_source_available_issues_exactly_one_warning(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            with warnings.catch_warnings(record=True) as all_warnings:
                eval('ic(a)')
                eval('ic(b)')
                assert len(all_warnings) == 1
                warning = all_warnings[-1]
                assert NO_SOURCE_AVAILABLE_WARNING_MESSAGE in str(warning.message)

    def test_single_tuple_argument(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic((a, b))

        pair = parse_output_into_pairs(out, err, 1)[0][0]
        self.assertEqual(pair, ('(a, b)', '(1, 2)'))

    def test_flat_medium_list_prints_on_one_line(self):
        """Flat medium-sized lists should not be split one item per line."""
        data = [1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
                1, 1, 1, 1, 1,
                0, 1, 0, 1, 0, 1, 0]

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)

        # The whole ic() call should fit on a single line.
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1)

    def test_multiline_container_args(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic((a,
                b))
            ic([a,
                b])
            ic((a,
                b),
               [list(range(15)),
                list(range(15))])

        self.assertEqual(err.getvalue().strip(), """
ic| (a,
     b): (1, 2)
ic| [a,
     b]: [1, 2]
ic| (a,
     b): (1, 2)
    [list(range(15)),
     list(range(15))]: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        """.strip())

        with disable_coloring(), capture_standard_streams() as (out, err):
            with configure_icecream_output(includeContext=True):
                ic((a,
                    b),
                   [list(range(15)),
                    list(range(15))])

        lines = err.getvalue().strip().splitlines()
        self.assertRegex(
            lines[0],
            r'ic\| test_icecream.py:\d+ in test_multiline_container_args\(\)',
        )
        self.assertEqual('\n'.join(lines[1:]), """\
    (a,
     b): (1, 2)
    [list(range(15)),
     list(range(15))]: [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]""")

    def test_multiple_tuple_arguments(self):
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic((a, b), (b, a), a, b)

        pair = parse_output_into_pairs(out, err, 1)[0]
        self.assertEqual(pair, [
            ('(a, b)', '(1, 2)'), ('(b, a)', '(2, 1)'), ('a', '1'), ('b', '2')])

    def test_coloring(self):
        with capture_standard_streams() as (out, err):
            ic({1: 'str'})  # Output should be colored with ANSI control codes.

        assert has_ansi_escape_codes(err.getvalue())

    def test_configure_output_with_no_parameters(self):
        with self.assertRaises(TypeError):
            ic.configureOutput()

    def test_multiline_strings_output(self):

        test1 = "A\\veryvery\\long\\path\\to\\no\\even\\longer\\HelloWorld _01_Heritisfinallythe file.file"
        test2 = r"A\veryvery\long\path\to\no\even\longer\HelloWorld _01_Heritisfinallythe file.file"
        test3 = "line\nline"

        with disable_coloring(), capture_standard_streams() as (_, err):
            ic(test1)
            curr_res = err.getvalue().strip()
            expected = r"ic| test1: 'A\\veryvery\\long\\path\\to\\no\\even\\longer\\HelloWorld _01_Heritisfinallythe file.file'"
            self.assertEqual(curr_res, expected)
            del curr_res, expected

        with disable_coloring(), capture_standard_streams() as (_, err):
            ic(test2)
            curr_res = err.getvalue().strip()
            # expected = r"ic| test2: 'A\\veryvery\\long\\path\\to\\no\\even\\longer\\HelloWorld _01_Heritisfinallythe file.file'"
            expected = r"ic| test2: 'A\\veryvery\\long\\path\\to\\no\\even\\longer\\HelloWorld _01_Heritisfinallythe file.file'"
            self.assertEqual(curr_res, expected)
            del curr_res, expected

        with disable_coloring(), capture_standard_streams() as (_, err):
            ic(test3)
            curr_res = err.getvalue().strip()
            expected = r"""ic| test3: '''line
            line'''"""
            self.assertEqual(curr_res, expected)
            del curr_res, expected

    def test_sympy_dict_keys_do_not_crash(self):
        """Regression: ic() must not raise when dict keys are SymPy symbols."""

        try:
            import sympy as sp
        except Exception:
            self.skipTest("sympy not installed")

        x, y = sp.symbols("x y")
        d = {x: "hello", y: "world"}

        with disable_coloring(), capture_standard_streams() as (out, err):
            # If the bug regresses, this line raises TypeError.
            ic(d)

        s = err.getvalue().strip()
        # Basic sanity checks without assuming exact formatting or ordering.
        self.assertIn("ic|", s)
        self.assertIn("hello", s)
        self.assertIn("world", s)

    def test_sympy_solve_result_does_not_crash(self):
        """Regression: ic() must handle SymPy solve() outputs."""

        try:
            import sympy as sp
        except Exception:
            self.skipTest("sympy not installed")

        x, y = sp.symbols("x y")
        res = sp.solve([x + 2, y - 2])   # list/dict of symbolic items

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(res)

        s = err.getvalue()
        self.assertIn("ic|", s)
        # Don't assert exact text; just ensure something printed.
        self.assertTrue(len(s) > 0)


class TestIndentation(unittest.TestCase):
    """Tests for the indentation feature for nested/recursive calls."""

    def setUp(self):
        ic._pairDelimiter = TEST_PAIR_DELIMITER
        # Ensure indentation is disabled by default
        ic.configureOutput(enableIndentation=False)
        ic.resetIndentation()

    def tearDown(self):
        # Reset to default state
        ic.configureOutput(enableIndentation=False)
        ic.resetIndentation()

    def test_indentation_disabled_by_default(self):
        """Indentation should be disabled by default."""
        self.assertFalse(ic.enableIndentation)

    def test_indentation_can_be_enabled(self):
        """Indentation can be enabled via configureOutput."""
        ic.configureOutput(enableIndentation=True)
        self.assertTrue(ic.enableIndentation)

    def test_no_indentation_when_disabled(self):
        """When indentation is disabled, all output should be left-aligned."""
        ic.configureOutput(enableIndentation=False)

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            return err.getvalue()

        def outer():
            return inner()

        result = outer()
        # Should start with ic| (no indentation)
        self.assertTrue(result.strip().startswith('ic|'))

    def test_indentation_with_nested_calls(self):
        """When indentation is enabled, nested calls should be indented."""
        ic.configureOutput(enableIndentation=True)
        ic.resetIndentation()

        results = []

        def level2():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(c)
            results.append(err.getvalue())

        def level1():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
            results.append(err.getvalue())
            level2()

        def level0():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            results.append(err.getvalue())
            level1()

        level0()

        # level0 should have no indentation (baseline)
        self.assertTrue(results[0].strip().startswith('ic|'))
        self.assertFalse(results[0].startswith('    '))

        # level1 should have 1 level of indentation
        self.assertTrue(results[1].startswith('    '))
        self.assertIn('ic|', results[1])

        # level2 should have 2 levels of indentation
        self.assertTrue(results[2].startswith('        '))
        self.assertIn('ic|', results[2])

    def test_recursive_function_indentation(self):
        """Recursive function calls should show increasing indentation."""
        ic.configureOutput(enableIndentation=True)
        ic.resetIndentation()

        results = []

        def recursive(n):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(n)
            results.append(err.getvalue())
            if n > 0:
                recursive(n - 1)

        recursive(2)

        # First call (n=2) should be baseline (no indentation)
        self.assertTrue(results[0].strip().startswith('ic|'))

        # Each subsequent call should have more indentation
        for i in range(1, len(results)):
            expected_indent = '    ' * i
            self.assertTrue(results[i].startswith(expected_indent),
                           f"Result {i} should start with {len(expected_indent)} spaces")

    def test_custom_indentation_string(self):
        """Custom indentation string can be configured."""
        ic.configureOutput(enableIndentation=True, indentationStr='--')
        ic.resetIndentation()

        results = []

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
            results.append(err.getvalue())

        def outer():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            results.append(err.getvalue())
            inner()

        outer()

        # Outer should have no indentation
        self.assertFalse(results[0].startswith('--'))

        # Inner should have '--' indentation
        self.assertTrue(results[1].startswith('--'))
        self.assertFalse(results[1].startswith('----'))

    def test_reset_indentation(self):
        """resetIndentation() should reset the baseline."""
        ic.configureOutput(enableIndentation=True)

        results = []

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
            results.append(err.getvalue())

        def outer():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            results.append(err.getvalue())
            inner()

        # First call sequence
        ic.resetIndentation()
        outer()

        # Reset and call again - should start fresh
        ic.resetIndentation()
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(c)
        results.append(err.getvalue())

        # The last call after reset should have no indentation
        self.assertTrue(results[-1].strip().startswith('ic|'))
        self.assertFalse(results[-1].startswith('    '))

    def test_multiline_output_indentation(self):
        """Multiline output should have all lines indented."""
        ic.configureOutput(enableIndentation=True)
        ic.resetIndentation()

        multilineStr = 'line1\nline2'

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(multilineStr)
            return err.getvalue()

        def outer():
            return inner()

        result = outer()
        lines = result.splitlines()

        # All lines should be indented
        for line in lines:
            self.assertTrue(line.startswith('    '),
                           f"Line '{line}' should be indented")

    def test_indentation_with_include_context(self):
        """Indentation should work with includeContext enabled."""
        ic.configureOutput(enableIndentation=True, includeContext=True)
        ic.resetIndentation()

        results = []

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
            results.append(err.getvalue())

        def outer():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            results.append(err.getvalue())
            inner()

        outer()

        # Reset includeContext
        ic.configureOutput(includeContext=False)

        # Outer should have no indentation but include context
        self.assertTrue(results[0].strip().startswith('ic|'))
        self.assertIn('in outer()', results[0])

        # Inner should be indented and include context
        self.assertTrue(results[1].startswith('    '))
        self.assertIn('in inner()', results[1])

    def test_disable_enable_preserves_baseline(self):
        """Disabling and re-enabling indentation should preserve the baseline."""
        ic.configureOutput(enableIndentation=True)
        ic.resetIndentation()

        results = []

        def level2():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(c)
            results.append(err.getvalue())

        def level1():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
            results.append(err.getvalue())
            level2()

        def level0():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            results.append(err.getvalue())
            level1()

        # Call level0 to establish baseline and capture indented output
        level0()

        # results[0] = level0 (baseline, no indent)
        # results[1] = level1 (1 level indent)
        # results[2] = level2 (2 levels indent)
        self.assertFalse(results[0].startswith('    '))
        self.assertTrue(results[1].startswith('    '))
        self.assertTrue(results[2].startswith('        '))

        # Now disable indentation
        ic.configureOutput(enableIndentation=False)

        # Call at same levels - should have no indentation when disabled
        results.clear()
        level0()
        self.assertFalse(results[0].startswith('    '))
        self.assertFalse(results[1].startswith('    '))
        self.assertFalse(results[2].startswith('    '))

        # Re-enable indentation - baseline should be preserved
        ic.configureOutput(enableIndentation=True)

        # Call at same levels - should have same indentation as before
        results.clear()
        level0()

        # Baseline was set at level0's stack depth, so:
        # level0 = no indent, level1 = 1 indent, level2 = 2 indents
        self.assertFalse(results[0].startswith('    '))
        self.assertTrue(results[1].startswith('    '))
        self.assertTrue(results[2].startswith('        '))


class TestSensitiveDataMasking(unittest.TestCase):
    """Tests for sensitive data masking feature."""

    def setUp(self):
        ic._pairDelimiter = TEST_PAIR_DELIMITER
        # Clear any sensitive keys from previous tests
        ic.configureSensitiveKeys(clear=True)

    def tearDown(self):
        # Reset sensitive keys after each test
        ic.configureSensitiveKeys(clear=True)

    def test_no_masking_by_default(self):
        """By default, no masking should occur."""
        password = 'secret123'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(password)
        self.assertIn('secret123', err.getvalue())

    def test_configure_sensitive_keys_requires_argument(self):
        """configureSensitiveKeys() should raise TypeError if no arguments."""
        with self.assertRaises(TypeError):
            ic.configureSensitiveKeys()

    def test_mask_simple_variable_by_name(self):
        """Variables with sensitive names should be masked."""
        ic.configureSensitiveKeys(keys=['password'])
        password = 'secret123'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(password)
        output = err.getvalue()
        self.assertNotIn('secret123', output)
        self.assertIn('***MASKED***', output)

    def test_mask_dict_values(self):
        """Dictionary values with sensitive keys should be masked."""
        ic.configureSensitiveKeys(keys=['password', 'api_key'])
        config = {'password': 'secret123', 'api_key': 'sk-abc123', 'debug': True}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(config)
        output = err.getvalue()
        self.assertNotIn('secret123', output)
        self.assertNotIn('sk-abc123', output)
        self.assertIn('***MASKED***', output)
        self.assertIn('True', output)  # Non-sensitive value should be visible

    def test_mask_nested_dict_values(self):
        """Nested dictionary values with sensitive keys should be masked."""
        ic.configureSensitiveKeys(keys=['password'])
        config = {'database': {'password': 'dbpass123', 'host': 'localhost'}}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(config)
        output = err.getvalue()
        self.assertNotIn('dbpass123', output)
        self.assertIn('***MASKED***', output)
        self.assertIn('localhost', output)

    def test_case_insensitive_matching(self):
        """Key matching should be case-insensitive."""
        ic.configureSensitiveKeys(keys=['password'])
        data = {'PASSWORD': 'upper', 'Password': 'mixed', 'password': 'lower'}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertNotIn('upper', output)
        self.assertNotIn('mixed', output)
        self.assertNotIn('lower', output)

    def test_substring_matching(self):
        """Key matching should use substring matching."""
        ic.configureSensitiveKeys(keys=['password'])
        data = {'user_password': 'pass1', 'passwordHash': 'pass2'}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertNotIn('pass1', output)
        self.assertNotIn('pass2', output)

    def test_mask_string_patterns(self):
        """Strings containing sensitive patterns should be masked.

        The string masking uses regex to find key-value patterns in the format:
            <sensitive_key><separator><value>

        Where:
            - <sensitive_key>: case-insensitive match of configured sensitive keys
            - <separator>: '=' or ':' with optional whitespace around it
            - <value>: characters until a delimiter (whitespace, comma, semicolon,
                       quotes, braces, brackets)

        Examples of patterns that will be masked:
            - 'password=secret123' -> 'password=***MASKED***'
            - 'token: abc123' -> 'token: ***MASKED***'
            - 'API_KEY=sk-123;other=val' -> 'API_KEY=***MASKED***;other=val'
        """
        ic.configureSensitiveKeys(keys=['password', 'token'])

        # Test semicolon-separated connection string
        connection_string = 'host=localhost;password=secret123;port=5432'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(connection_string)
        output = err.getvalue()
        self.assertNotIn('secret123', output)
        self.assertIn('***MASKED***', output)
        self.assertIn('localhost', output)

    def test_mask_string_patterns_with_colon_separator(self):
        """String patterns with colon separator should be masked."""
        ic.configureSensitiveKeys(keys=['token'])
        config_line = 'token: mytoken123, debug: true'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(config_line)
        output = err.getvalue()
        self.assertNotIn('mytoken123', output)
        self.assertIn('***MASKED***', output)
        self.assertIn('debug', output)

    def test_mask_string_patterns_with_whitespace(self):
        """String patterns with whitespace around separator should be masked."""
        ic.configureSensitiveKeys(keys=['password'])
        config_line = 'password = mysecret'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(config_line)
        output = err.getvalue()
        self.assertNotIn('mysecret', output)
        self.assertIn('***MASKED***', output)

    def test_mask_list_of_dicts(self):
        """Lists containing dicts with sensitive keys should be masked."""
        ic.configureSensitiveKeys(keys=['token'])
        users = [{'name': 'alice', 'token': 'tok1'}, {'name': 'bob', 'token': 'tok2'}]
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(users)
        output = err.getvalue()
        self.assertNotIn('tok1', output)
        self.assertNotIn('tok2', output)
        self.assertIn('alice', output)
        self.assertIn('bob', output)

    def test_custom_placeholder(self):
        """Custom placeholder should be used when configured."""
        ic.configureSensitiveKeys(keys=['password'], placeholder='[REDACTED]')
        password = 'secret123'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(password)
        output = err.getvalue()
        self.assertNotIn('secret123', output)
        self.assertIn('[REDACTED]', output)

    def test_add_keys(self):
        """Keys can be added incrementally."""
        ic.configureSensitiveKeys(keys=['password'])
        ic.configureSensitiveKeys(add=['api_key'])

        data = {'password': 'pass1', 'api_key': 'key1'}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertNotIn('pass1', output)
        self.assertNotIn('key1', output)

    def test_remove_keys(self):
        """Keys can be removed from masking."""
        ic.configureSensitiveKeys(keys=['password', 'api_key'])
        ic.configureSensitiveKeys(remove=['password'])

        data = {'password': 'pass1', 'api_key': 'key1'}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertIn('pass1', output)  # Should NOT be masked
        self.assertNotIn('key1', output)  # Should be masked

    def test_clear_keys(self):
        """All keys can be cleared."""
        ic.configureSensitiveKeys(keys=['password'])
        ic.configureSensitiveKeys(clear=True)

        password = 'secret123'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(password)
        self.assertIn('secret123', err.getvalue())

    def test_sensitive_keys_property(self):
        """sensitiveKeys property should return current keys."""
        ic.configureSensitiveKeys(keys=['password', 'token'])
        keys = ic.sensitiveKeys
        self.assertEqual(keys, {'password', 'token'})

    def test_sensitive_placeholder_property(self):
        """sensitivePlaceholder property should return current placeholder."""
        ic.configureSensitiveKeys(keys=['test'], placeholder='[HIDDEN]')
        self.assertEqual(ic.sensitivePlaceholder, '[HIDDEN]')

    def test_mask_tuple_values(self):
        """Tuples containing dicts with sensitive keys should be masked."""
        ic.configureSensitiveKeys(keys=['secret'])
        data = ({'secret': 'val1'}, {'public': 'val2'})
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertNotIn('val1', output)
        self.assertIn('val2', output)

    def test_non_sensitive_data_unchanged(self):
        """Non-sensitive data should remain unchanged."""
        ic.configureSensitiveKeys(keys=['password'])
        data = {'username': 'alice', 'email': 'alice@example.com', 'age': 30}
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertIn('alice', output)
        self.assertIn('alice@example.com', output)
        self.assertIn('30', output)

    def test_multiple_sensitive_keys(self):
        """Multiple sensitive keys should all be masked."""
        ic.configureSensitiveKeys(keys=['password', 'api_key', 'token', 'secret'])
        data = {
            'password': 'pass123',
            'api_key': 'key456',
            'token': 'tok789',
            'secret': 'sec000',
            'public': 'visible'
        }
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(data)
        output = err.getvalue()
        self.assertNotIn('pass123', output)
        self.assertNotIn('key456', output)
        self.assertNotIn('tok789', output)
        self.assertNotIn('sec000', output)
        self.assertIn('visible', output)

    def test_format_method_also_masks(self):
        """The format() method should also mask sensitive values."""
        ic.configureSensitiveKeys(keys=['password'])
        password = 'secret123'
        result = ic.format(password)
        self.assertNotIn('secret123', result)
        self.assertIn('***MASKED***', result)


class TestSensitiveKeyHelpers(unittest.TestCase):
    """Tests for the module-level helper functions for sensitive key masking."""

    def test_is_sensitive_key_exact_match(self):
        """is_sensitive_key should match exact key names."""
        self.assertTrue(is_sensitive_key('password', {'password'}))
        self.assertTrue(is_sensitive_key('api_key', {'api_key'}))

    def test_is_sensitive_key_case_insensitive(self):
        """is_sensitive_key should be case-insensitive."""
        self.assertTrue(is_sensitive_key('PASSWORD', {'password'}))
        self.assertTrue(is_sensitive_key('password', {'PASSWORD'}))
        self.assertTrue(is_sensitive_key('PaSsWoRd', {'password'}))

    def test_is_sensitive_key_substring_match(self):
        """is_sensitive_key should match substrings."""
        self.assertTrue(is_sensitive_key('user_password', {'password'}))
        self.assertTrue(is_sensitive_key('passwordHash', {'password'}))
        self.assertTrue(is_sensitive_key('db_password_encrypted', {'password'}))

    def test_is_sensitive_key_no_match(self):
        """is_sensitive_key should return False for non-matching keys."""
        self.assertFalse(is_sensitive_key('username', {'password'}))
        self.assertFalse(is_sensitive_key('email', {'password', 'token'}))

    def test_is_sensitive_key_empty_sensitive_keys(self):
        """is_sensitive_key should return False for empty sensitive keys set."""
        self.assertFalse(is_sensitive_key('password', set()))

    def test_mask_sensitive_value_dict(self):
        """_mask_sensitive_value should mask dict values with sensitive keys."""
        data = {'password': 'secret', 'username': 'alice'}
        result = _mask_sensitive_value(data, {'password'}, '***')
        self.assertEqual(result, {'password': '***', 'username': 'alice'})

    def test_mask_sensitive_value_nested_dict(self):
        """_mask_sensitive_value should mask nested dict values."""
        data = {'config': {'password': 'secret', 'host': 'localhost'}}
        result = _mask_sensitive_value(data, {'password'}, '***')
        self.assertEqual(result, {'config': {'password': '***', 'host': 'localhost'}})

    def test_mask_sensitive_value_list(self):
        """_mask_sensitive_value should mask values in lists."""
        data = [{'password': 'pass1'}, {'password': 'pass2'}]
        result = _mask_sensitive_value(data, {'password'}, '***')
        self.assertEqual(result, [{'password': '***'}, {'password': '***'}])

    def test_mask_sensitive_value_string_pattern(self):
        """_mask_sensitive_value should mask patterns in strings."""
        data = 'password=secret123;host=localhost'
        result = _mask_sensitive_value(data, {'password'}, '***')
        self.assertEqual(result, 'password=***;host=localhost')

    def test_mask_sensitive_value_empty_sensitive_keys(self):
        """_mask_sensitive_value should return unchanged object for empty keys."""
        data = {'password': 'secret'}
        result = _mask_sensitive_value(data, set(), '***')
        self.assertEqual(result, data)

    def test_mask_sensitive_value_preserves_non_sensitive(self):
        """_mask_sensitive_value should preserve non-sensitive values."""
        data = {'public': 'visible', 'count': 42, 'active': True}
        result = _mask_sensitive_value(data, {'password'}, '***')
        self.assertEqual(result, data)


class TestCallLimiting(unittest.TestCase):
    """Tests for the call limiting feature."""

    def setUp(self):
        ic._pairDelimiter = TEST_PAIR_DELIMITER
        # Reset all call limit settings
        ic.configureOutput(limitFirst=None, limitLast=None, limitEvery=None)
        ic.resetCallLimit()

    def tearDown(self):
        # Reset to default state
        ic.configureOutput(limitFirst=None, limitLast=None, limitEvery=None)
        ic.resetCallLimit()

    def test_no_limits_by_default(self):
        """By default, no limits should be set."""
        self.assertIsNone(ic.limitFirst)
        self.assertIsNone(ic.limitLast)
        self.assertIsNone(ic.limitEvery)

    def test_call_count_increments(self):
        """Call count should increment with each ic() call."""
        ic.resetCallLimit()
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)
            ic(b)
            ic(c)
        self.assertEqual(ic.callCount, 3)

    def test_output_count_tracks_outputs(self):
        """Output count should track how many calls produced output."""
        ic.resetCallLimit()
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)
            ic(b)
        self.assertEqual(ic.outputCount, 2)

    def test_limit_first_shows_only_first_n(self):
        """limitFirst=N should only show the first N ic() calls."""
        ic.configureOutput(limitFirst=2)
        ic.resetCallLimit()

        outputs = []
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Should output
            ic(2)  # Should output
            ic(3)  # Should be suppressed
            ic(4)  # Should be suppressed

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn('1', lines[0])
        self.assertIn('2', lines[1])
        self.assertEqual(ic.callCount, 4)
        self.assertEqual(ic.outputCount, 2)

    def test_limit_first_zero_suppresses_all(self):
        """limitFirst=0 should suppress all output."""
        ic.configureOutput(limitFirst=0)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)
            ic(3)

        self.assertEqual(err.getvalue(), '')
        self.assertEqual(ic.callCount, 3)
        self.assertEqual(ic.outputCount, 0)

    def test_limit_every_shows_every_nth(self):
        """limitEvery=N should show every Nth call (1st, N+1th, 2N+1th, ...)."""
        ic.configureOutput(limitEvery=3)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Call 1: output (1st)
            ic(2)  # Call 2: suppress
            ic(3)  # Call 3: suppress
            ic(4)  # Call 4: output (3+1=4th)
            ic(5)  # Call 5: suppress
            ic(6)  # Call 6: suppress
            ic(7)  # Call 7: output (6+1=7th)

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn('1', lines[0])
        self.assertIn('4', lines[1])
        self.assertIn('7', lines[2])
        self.assertEqual(ic.callCount, 7)
        self.assertEqual(ic.outputCount, 3)

    def test_limit_every_one_shows_all(self):
        """limitEvery=1 should show all calls."""
        ic.configureOutput(limitEvery=1)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)
            ic(3)

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual(ic.outputCount, 3)

    def test_limit_first_and_every_combined(self):
        """limitFirst and limitEvery can be combined (both conditions must be met)."""
        ic.configureOutput(limitFirst=5, limitEvery=2)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Call 1: matches every=2 (1st), within first=5 -> output
            ic(2)  # Call 2: doesn't match every=2 -> suppress
            ic(3)  # Call 3: matches every=2 (3rd), within first=5 -> output
            ic(4)  # Call 4: doesn't match every=2 -> suppress
            ic(5)  # Call 5: matches every=2 (5th), within first=5 -> output
            ic(6)  # Call 6: exceeds first=5 -> suppress
            ic(7)  # Call 7: exceeds first=5 -> suppress

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 3)
        self.assertIn('1', lines[0])
        self.assertIn('3', lines[1])
        self.assertIn('5', lines[2])

    def test_limit_last_buffers_output(self):
        """limitLast=N should buffer calls and only flush last N."""
        ic.configureOutput(limitLast=2)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Buffered, then dropped
            ic(2)  # Buffered, then dropped
            ic(3)  # Buffered, kept
            ic(4)  # Buffered, kept

        # Nothing should be output yet
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(ic.callCount, 4)

        # Now flush
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn('3', lines[0])
        self.assertIn('4', lines[1])

    def test_limit_last_with_fewer_calls_than_limit(self):
        """limitLast=N with fewer than N calls should show all calls on flush."""
        ic.configureOutput(limitLast=5)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)

        # Flush
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_reset_call_limit_clears_counters(self):
        """resetCallLimit() should clear counters and buffer."""
        ic.configureOutput(limitLast=5)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)

        self.assertEqual(ic.callCount, 2)

        ic.resetCallLimit()

        self.assertEqual(ic.callCount, 0)
        self.assertEqual(ic.outputCount, 0)

        # Buffer should also be cleared
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        self.assertEqual(err.getvalue(), '')

    def test_flush_clears_buffer(self):
        """flushCallLimit() should clear the buffer after flushing."""
        ic.configureOutput(limitLast=3)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)

        # First flush
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        lines = err.getvalue().strip().split('\n')
        self.assertEqual(len([l for l in lines if l.strip()]), 2)

        # Second flush should produce nothing (buffer cleared)
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        self.assertEqual(err.getvalue(), '')

    def test_limit_none_disables_limit(self):
        """Setting limit to None should disable that limit."""
        ic.configureOutput(limitFirst=2)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)
            ic(3)

        self.assertEqual(ic.outputCount, 2)

        # Disable the limit
        ic.configureOutput(limitFirst=None)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)
            ic(3)

        self.assertEqual(ic.outputCount, 3)

    def test_call_counting_with_disabled_ic(self):
        """Call count should not increment when ic is disabled."""
        ic.resetCallLimit()
        ic.disable()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)
            ic(2)

        self.assertEqual(ic.callCount, 0)

        ic.enable()
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(3)

        self.assertEqual(ic.callCount, 1)

    def test_limit_first_with_indentation(self):
        """limitFirst should work correctly with indentation enabled."""
        ic.configureOutput(enableIndentation=True, limitFirst=2)
        ic.resetIndentation()
        ic.resetCallLimit()

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)
            return err.getvalue()

        def outer():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            result1 = err.getvalue()
            result2 = inner()
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(c)  # This should be suppressed (3rd call)
            result3 = err.getvalue()
            return result1, result2, result3

        r1, r2, r3 = outer()

        # First two calls should have output
        self.assertIn('ic|', r1)
        self.assertIn('ic|', r2)
        # Third call should be suppressed
        self.assertEqual(r3, '')
        self.assertEqual(ic.callCount, 3)
        self.assertEqual(ic.outputCount, 2)

        # Reset
        ic.configureOutput(enableIndentation=False)

    def test_limit_last_with_indentation(self):
        """limitLast should preserve indentation in buffered output."""
        ic.configureOutput(enableIndentation=True, limitLast=2)
        ic.resetIndentation()
        ic.resetCallLimit()

        def inner():
            ic(b)

        def outer():
            ic(a)  # Call 1 - baseline (no indent)
            inner()  # Call 2 - one level deep (indented)
            ic(c)  # Call 3 - back to baseline (no indent)

        with disable_coloring(), capture_standard_streams() as (out, err):
            outer()

        # Nothing should be output yet (buffered)
        self.assertEqual(err.getvalue(), '')

        # Flush and check
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = output.strip().split('\n')
        # Should have last 2 calls (inner's ic(b) and ic(c))
        # The indentation should be preserved
        self.assertEqual(len([l for l in lines if l.strip()]), 2)

        # First buffered line should be from inner() - indented
        self.assertTrue(lines[0].startswith('    '),
                       f"Expected inner() call to be indented, got: '{lines[0]}'")
        self.assertIn('b', lines[0])

        # Second buffered line should be from outer()'s second ic(c) - not indented
        self.assertFalse(lines[1].startswith('    '),
                        f"Expected outer() call to not be indented, got: '{lines[1]}'")
        self.assertIn('c', lines[1])

        # Reset
        ic.configureOutput(enableIndentation=False)

    def test_limit_last_with_deep_nesting_preserves_indentation(self):
        """limitLast should correctly preserve multiple indentation levels."""
        ic.configureOutput(enableIndentation=True, limitLast=4)
        ic.resetIndentation()
        ic.resetCallLimit()

        def level3():
            ic('L3')  # 3 levels deep

        def level2():
            ic('L2')  # 2 levels deep
            level3()

        def level1():
            ic('L1')  # 1 level deep
            level2()

        def level0():
            ic('L0')  # baseline
            level1()

        with disable_coloring(), capture_standard_streams() as (out, err):
            level0()

        # Nothing should be output yet
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(ic.callCount, 4)

        # Flush and verify indentation levels
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = output.strip().split('\n')
        self.assertEqual(len([l for l in lines if l.strip()]), 4)

        # L0 - baseline (no indentation)
        self.assertFalse(lines[0].startswith(' '),
                        f"L0 should have no indentation: '{lines[0]}'")
        self.assertIn('L0', lines[0])

        # L1 - 1 level (4 spaces)
        self.assertTrue(lines[1].startswith('    ') and not lines[1].startswith('        '),
                       f"L1 should have 1 level indentation: '{lines[1]}'")
        self.assertIn('L1', lines[1])

        # L2 - 2 levels (8 spaces)
        self.assertTrue(lines[2].startswith('        ') and not lines[2].startswith('            '),
                       f"L2 should have 2 levels indentation: '{lines[2]}'")
        self.assertIn('L2', lines[2])

        # L3 - 3 levels (12 spaces)
        self.assertTrue(lines[3].startswith('            '),
                       f"L3 should have 3 levels indentation: '{lines[3]}'")
        self.assertIn('L3', lines[3])

        # Reset
        ic.configureOutput(enableIndentation=False)

    def test_limit_last_with_recursive_preserves_indentation(self):
        """limitLast should preserve indentation in recursive calls."""
        ic.configureOutput(enableIndentation=True, limitLast=3)
        ic.resetIndentation()
        ic.resetCallLimit()

        def recursive(n):
            ic(n)
            if n > 0:
                recursive(n - 1)

        with disable_coloring(), capture_standard_streams() as (out, err):
            recursive(4)  # Makes 5 calls (n=4,3,2,1,0)

        # Nothing output yet
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(ic.callCount, 5)

        # Flush - should show last 3 calls (n=2,1,0) with increasing indentation
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = output.strip().split('\n')
        self.assertEqual(len([l for l in lines if l.strip()]), 3)

        # The last 3 calls are at depths 2, 3, 4 relative to baseline
        # n=2 at depth 2 (8 spaces)
        self.assertTrue(lines[0].startswith('        '),
                       f"n=2 should have 2 levels indentation: '{lines[0]}'")
        self.assertIn('2', lines[0])

        # n=1 at depth 3 (12 spaces)
        self.assertTrue(lines[1].startswith('            '),
                       f"n=1 should have 3 levels indentation: '{lines[1]}'")
        self.assertIn('1', lines[1])

        # n=0 at depth 4 (16 spaces)
        self.assertTrue(lines[2].startswith('                '),
                       f"n=0 should have 4 levels indentation: '{lines[2]}'")
        self.assertIn('0', lines[2])

        # Reset
        ic.configureOutput(enableIndentation=False)

    def test_limit_last_with_custom_indentation_string(self):
        """limitLast should preserve custom indentation strings."""
        ic.configureOutput(enableIndentation=True, indentationStr='>>',  limitLast=2)
        ic.resetIndentation()
        ic.resetCallLimit()

        def inner():
            ic('inner')

        def outer():
            ic('outer')
            inner()

        with disable_coloring(), capture_standard_streams() as (out, err):
            outer()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = output.strip().split('\n')

        # outer() call - no indentation
        self.assertFalse(lines[0].startswith('>>'),
                        f"outer() should have no indentation: '{lines[0]}'")

        # inner() call - custom indentation '>>'
        self.assertTrue(lines[1].startswith('>>'),
                       f"inner() should start with '>>': '{lines[1]}'")
        self.assertFalse(lines[1].startswith('>>>>'),
                        f"inner() should only have 1 level: '{lines[1]}'")

        # Reset
        ic.configureOutput(enableIndentation=False, indentationStr='    ')

    def test_properties_reflect_configuration(self):
        """Property accessors should reflect current configuration."""
        ic.configureOutput(limitFirst=10, limitLast=5, limitEvery=2)

        self.assertEqual(ic.limitFirst, 10)
        self.assertEqual(ic.limitLast, 5)
        self.assertEqual(ic.limitEvery, 2)

        ic.configureOutput(limitFirst=None, limitLast=None, limitEvery=None)

        self.assertIsNone(ic.limitFirst)
        self.assertIsNone(ic.limitLast)
        self.assertIsNone(ic.limitEvery)

    def test_limit_every_large_value(self):
        """limitEvery with large value should work correctly."""
        ic.configureOutput(limitEvery=100)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            for i in range(10):
                ic(i)

        # Only the first call (call 1) should output
        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertIn('0', lines[0])

    def test_recursive_function_with_limit_first(self):
        """limitFirst should work with recursive functions."""
        ic.configureOutput(limitFirst=3)
        ic.resetCallLimit()

        def recursive(n):
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(n)
            result = err.getvalue()
            if n > 0:
                recursive(n - 1)
            return result

        # This will make 6 calls (n=5,4,3,2,1,0)
        outputs = []

        def capture_recursive(n):
            if n >= 0:
                with disable_coloring(), capture_standard_streams() as (out, err):
                    ic(n)
                outputs.append(err.getvalue())
                capture_recursive(n - 1)

        capture_recursive(5)

        # Only first 3 should have output
        non_empty = [o for o in outputs if o.strip()]
        self.assertEqual(len(non_empty), 3)
        self.assertEqual(ic.callCount, 6)

    def test_loop_with_limit_every(self):
        """limitEvery should be useful for loops."""
        ic.configureOutput(limitEvery=10)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            for i in range(100):
                ic(i)

        # Should output calls 1, 11, 21, 31, 41, 51, 61, 71, 81, 91 = 10 outputs
        # (indices 0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]
        self.assertEqual(len(lines), 10)
        self.assertEqual(ic.outputCount, 10)
        self.assertEqual(ic.callCount, 100)

    def test_limit_first_with_include_context(self):
        """limitFirst should work correctly when includeContext is enabled."""
        ic.configureOutput(limitFirst=3, includeContext=True)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Call 1 - should output with context
            ic(2)  # Call 2 - should output with context
            ic(3)  # Call 3 - should output with context
            ic(4)  # Call 4 - should be suppressed
            ic(5)  # Call 5 - should be suppressed

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]

        # Should have 3 outputs
        self.assertEqual(ic.callCount, 5)
        self.assertEqual(ic.outputCount, 3)

        # Each output should include context info (filename and line number)
        # Context format: "ic| test_icecream.py:XXXX in test_limit_first_with_include_context()"
        for line in lines:
            self.assertIn('test_icecream.py:', line,
                         f"Context should include filename: '{line}'")
            self.assertIn('in test_limit_first_with_include_context()', line,
                         f"Context should include function name: '{line}'")

        # Verify values are correct (1, 2, 3)
        self.assertIn('1', lines[0])
        self.assertIn('2', lines[1])
        self.assertIn('3', lines[2])

        # Reset
        ic.configureOutput(includeContext=False)

    def test_limit_every_with_include_context(self):
        """limitEvery should work correctly when includeContext is enabled."""
        ic.configureOutput(limitEvery=3, includeContext=True)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            for i in range(9):
                ic(i)

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]

        # Should output calls 1, 4, 7 (i=0, 3, 6)
        self.assertEqual(ic.callCount, 9)
        self.assertEqual(ic.outputCount, 3)

        # Each output should include context
        for line in lines:
            self.assertIn('test_icecream.py:', line)

        # Verify correct values
        self.assertIn('0', lines[0])
        self.assertIn('3', lines[1])
        self.assertIn('6', lines[2])

        # Reset
        ic.configureOutput(includeContext=False)

    def test_disable_enable_mid_loop_call_counter(self):
        """Call counter should only count when ic is enabled."""
        ic.configureOutput(limitFirst=5)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            # First 3 iterations with ic enabled
            for i in range(3):
                ic(i)  # Calls 1, 2, 3

            # Disable ic for next 4 iterations
            ic.disable()
            for i in range(3, 7):
                ic(i)  # Should not count (ic disabled)

            # Re-enable ic for remaining iterations
            ic.enable()
            for i in range(7, 12):
                ic(i)  # Calls 4, 5, 6, 7, 8

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]

        # Call counter: 3 (enabled) + 0 (disabled) + 5 (enabled) = 8
        self.assertEqual(ic.callCount, 8,
                        "Call counter should only count enabled calls")

        # With limitFirst=5, should output calls 1-5 (i=0,1,2,7,8)
        # Calls 6,7,8 (i=9,10,11) should be suppressed
        self.assertEqual(ic.outputCount, 5,
                        "Should output only first 5 enabled calls")
        self.assertEqual(len(lines), 5)

        # Verify the output values: 0, 1, 2 from first batch, then 7, 8 from third batch
        self.assertIn('0', lines[0])
        self.assertIn('1', lines[1])
        self.assertIn('2', lines[2])
        self.assertIn('7', lines[3])
        self.assertIn('8', lines[4])

    def test_disable_enable_mid_loop_with_limit_every(self):
        """limitEvery should correctly track calls across disable/enable cycles."""
        ic.configureOutput(limitEvery=2)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            # Enabled: calls 1, 2, 3 (outputs 1, 3)
            ic('a')  # Call 1 - output
            ic('b')  # Call 2 - skip
            ic('c')  # Call 3 - output

            # Disabled: no counting
            ic.disable()
            ic('x')  # Not counted
            ic('y')  # Not counted

            # Enabled: calls 4, 5, 6 (outputs 5)
            ic.enable()
            ic('d')  # Call 4 - skip
            ic('e')  # Call 5 - output
            ic('f')  # Call 6 - skip

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]

        self.assertEqual(ic.callCount, 6,
                        "Should count 6 enabled calls")
        # limitEvery=2: outputs calls 1, 3, 5 (every odd call)
        self.assertEqual(ic.outputCount, 3)
        self.assertEqual(len(lines), 3)

        self.assertIn('a', lines[0])
        self.assertIn('c', lines[1])
        self.assertIn('e', lines[2])

    def test_disable_enable_with_limit_last(self):
        """limitLast should only buffer enabled calls."""
        ic.configureOutput(limitLast=3)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic('a')  # Call 1 - buffered
            ic('b')  # Call 2 - buffered

            ic.disable()
            ic('x')  # Not buffered (disabled)
            ic('y')  # Not buffered (disabled)

            ic.enable()
            ic('c')  # Call 3 - buffered
            ic('d')  # Call 4 - buffered (pushes 'a' out)
            ic('e')  # Call 5 - buffered (pushes 'b' out)

        # Nothing output yet
        self.assertEqual(err.getvalue(), '')
        self.assertEqual(ic.callCount, 5)

        # Flush - should show last 3 enabled calls: c, d, e
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        output = err.getvalue()
        lines = [l for l in output.strip().split('\n') if l.strip()]

        self.assertEqual(len(lines), 3)
        self.assertIn('c', lines[0])
        self.assertIn('d', lines[1])
        self.assertIn('e', lines[2])

    def test_multiple_disable_enable_cycles(self):
        """Call counter should work correctly through multiple disable/enable cycles."""
        ic.configureOutput(limitFirst=None, limitEvery=None)
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            # Cycle 1: enabled
            ic(1)
            ic(2)

            # Cycle 2: disabled
            ic.disable()
            ic(3)
            ic(4)

            # Cycle 3: enabled
            ic.enable()
            ic(5)

            # Cycle 4: disabled
            ic.disable()
            ic(6)

            # Cycle 5: enabled
            ic.enable()
            ic(7)
            ic(8)
            ic(9)

        self.assertEqual(ic.callCount, 6,
                        "Should count only enabled calls: 1,2,5,7,8,9")
        self.assertEqual(ic.outputCount, 6,
                        "With no limits, all enabled calls should output")


class TestFileLogging(unittest.TestCase):
    """Tests for the file logging feature."""

    def setUp(self):
        ic._pairDelimiter = TEST_PAIR_DELIMITER
        # Ensure logging is disabled
        ic.configureOutput(logFile=None)

    def tearDown(self):
        # Disable logging after each test
        ic.configureOutput(logFile=None)
        # Clean up any test log files
        import os
        for f in ['test_log.log', 'test_append.log', 'test_overwrite.log',
                  'test_context.log', 'test_limit.log', 'test_multiline.log',
                  'test_indent.log', 'test_ansi.log']:
            if os.path.exists(f):
                os.remove(f)

    def test_logging_disabled_by_default(self):
        """File logging should be disabled by default."""
        self.assertFalse(ic.isLogging)
        self.assertIsNone(ic.logFilePath)

    def test_enable_logging_creates_file(self):
        """configureOutput(logFile=...) should create a log file."""
        ic.configureOutput(logFile='test_log.log')
        self.assertTrue(ic.isLogging)
        self.assertEqual(ic.logFilePath, 'test_log.log')
        import os
        self.assertTrue(os.path.exists('test_log.log'))

    def test_disable_logging_closes_file(self):
        """configureOutput(logFile=None) should close the log file."""
        ic.configureOutput(logFile='test_log.log')
        self.assertTrue(ic.isLogging)
        ic.configureOutput(logFile=None)
        self.assertFalse(ic.isLogging)
        self.assertIsNone(ic.logFilePath)

    def test_basic_logging(self):
        """ic() output should be written to the log file."""
        ic.configureOutput(logFile='test_log.log')

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)
            ic(b)

        ic.configureOutput(logFile=None)

        with open('test_log.log', 'r') as f:
            content = f.read()

        self.assertIn('a: 1', content)
        self.assertIn('b: 2', content)

    def test_logging_strips_ansi_codes(self):
        """Log file should not contain ANSI escape codes."""
        ic.configureOutput(logFile='test_ansi.log')

        with capture_standard_streams() as (out, err):
            ic({1: 'test'})  # Colorized output

        ic.configureOutput(logFile=None)

        with open('test_ansi.log', 'r') as f:
            content = f.read()

        # ANSI escape codes start with \x1b[
        self.assertNotIn('\x1b[', content)
        # But the content should still be there
        self.assertIn('1', content)
        self.assertIn('test', content)

    def test_logging_append_mode(self):
        """Append mode ('a') should add to existing file content."""
        # First logging session
        ic.configureOutput(logFile='test_append.log', logMode='a')
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)
        ic.configureOutput(logFile=None)

        # Second logging session (append)
        ic.configureOutput(logFile='test_append.log', logMode='a')
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(b)
        ic.configureOutput(logFile=None)

        with open('test_append.log', 'r') as f:
            content = f.read()

        # Both outputs should be present
        self.assertIn('a: 1', content)
        self.assertIn('b: 2', content)

    def test_logging_overwrite_mode(self):
        """Overwrite mode ('w') should replace existing file content."""
        # First logging session
        ic.configureOutput(logFile='test_overwrite.log', logMode='a')
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)
        ic.configureOutput(logFile=None)

        # Second logging session (overwrite)
        ic.configureOutput(logFile='test_overwrite.log', logMode='w')
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(b)
        ic.configureOutput(logFile=None)

        with open('test_overwrite.log', 'r') as f:
            content = f.read()

        # Only second output should be present
        self.assertNotIn('a: 1', content)
        self.assertIn('b: 2', content)

    def test_logging_with_configure_output(self):
        """configureOutput should enable/disable logging."""
        ic.configureOutput(logFile='test_log.log')
        self.assertTrue(ic.isLogging)

        ic.configureOutput(logFile=None)
        self.assertFalse(ic.isLogging)

    def test_logging_with_context(self):
        """Logging should work with includeContext enabled."""
        ic.configureOutput(includeContext=True, logFile='test_context.log')

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)

        ic.configureOutput(logFile=None)
        ic.configureOutput(includeContext=False)

        with open('test_context.log', 'r') as f:
            content = f.read()

        self.assertIn('a: 1', content)
        self.assertIn('test_icecream.py:', content)
        self.assertIn('in test_logging_with_context()', content)

    def test_logging_with_limit_first(self):
        """Logging should respect limitFirst setting."""
        ic.configureOutput(limitFirst=2, logFile='test_limit.log')
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Should log
            ic(2)  # Should log
            ic(3)  # Should NOT log (exceeds limit)

        ic.configureOutput(logFile=None)
        ic.configureOutput(limitFirst=None)

        with open('test_limit.log', 'r') as f:
            content = f.read()

        self.assertIn('1', content)
        self.assertIn('2', content)
        self.assertNotIn('3', content)

    def test_logging_with_limit_last_flush(self):
        """Logging should work when flushing limitLast buffer."""
        ic.configureOutput(limitLast=2, logFile='test_limit.log')
        ic.resetCallLimit()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(1)  # Buffered, then dropped
            ic(2)  # Buffered, then dropped
            ic(3)  # Buffered, kept
            ic(4)  # Buffered, kept

        # Nothing logged yet (buffered)
        with open('test_limit.log', 'r') as f:
            content = f.read()
        self.assertEqual(content, '')

        # Flush - should log last 2
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic.flushCallLimit()

        ic.configureOutput(logFile=None)
        ic.configureOutput(limitLast=None)

        with open('test_limit.log', 'r') as f:
            content = f.read()

        self.assertNotIn('1', content)
        self.assertNotIn('2', content)
        self.assertIn('3', content)
        self.assertIn('4', content)

    def test_logging_multiline_output(self):
        """Multiline output should be logged correctly."""
        ic.configureOutput(logFile='test_multiline.log')

        multilineStr = 'line1\nline2\nline3'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(multilineStr)

        ic.configureOutput(logFile=None)

        with open('test_multiline.log', 'r') as f:
            content = f.read()

        self.assertIn('multilineStr', content)
        self.assertIn('line1', content)
        self.assertIn('line2', content)
        self.assertIn('line3', content)

    def test_logging_with_indentation(self):
        """Logging should preserve indentation."""
        ic.configureOutput(enableIndentation=True, logFile='test_indent.log')
        ic.resetIndentation()

        def inner():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(b)

        def outer():
            with disable_coloring(), capture_standard_streams() as (out, err):
                ic(a)
            inner()

        outer()

        ic.configureOutput(logFile=None)
        ic.configureOutput(enableIndentation=False)

        with open('test_indent.log', 'r') as f:
            lines = f.readlines()

        # First line should not be indented
        self.assertTrue(lines[0].strip().startswith('ic|'))
        # Second line should be indented
        self.assertTrue(lines[1].startswith('    '))

    def test_log_mode_property(self):
        """logMode property should reflect current mode."""
        ic.configureOutput(logFile='test_log.log', logMode='a')
        self.assertEqual(ic.logMode, 'a')
        ic.configureOutput(logFile=None)

        ic.configureOutput(logFile='test_log.log', logMode='w')
        self.assertEqual(ic.logMode, 'w')
        ic.configureOutput(logFile=None)

    def test_logging_output_also_goes_to_stderr(self):
        """Logging should not prevent normal output to stderr."""
        ic.configureOutput(logFile='test_log.log')

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)

        stderr_output = err.getvalue()

        ic.configureOutput(logFile=None)

        # Should also output to stderr
        self.assertIn('a: 1', stderr_output)

    def test_switching_log_files(self):
        """Switching to a new log file should close the old one."""
        ic.configureOutput(logFile='test_log.log')
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)

        ic.configureOutput(logFile='test_append.log')  # Switch to new file
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(b)

        ic.configureOutput(logFile=None)

        with open('test_log.log', 'r') as f:
            content1 = f.read()
        with open('test_append.log', 'r') as f:
            content2 = f.read()

        self.assertIn('a: 1', content1)
        self.assertNotIn('b: 2', content1)
        self.assertIn('b: 2', content2)
        self.assertNotIn('a: 1', content2)

    def test_logging_with_ic_disabled(self):
        """When ic is disabled, nothing should be logged."""
        ic.configureOutput(logFile='test_log.log')
        ic.disable()

        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(a)

        ic.enable()
        ic.configureOutput(logFile=None)

        with open('test_log.log', 'r') as f:
            content = f.read()

        self.assertEqual(content, '')

    def test_logging_with_sensitive_data_masking(self):
        """Sensitive data should be masked in log output."""
        ic.configureSensitiveKeys(keys=['password'])
        ic.configureOutput(logFile='test_log.log')

        password = 'supersecret123'
        with disable_coloring(), capture_standard_streams() as (out, err):
            ic(password)

        ic.configureOutput(logFile=None)
        ic.configureSensitiveKeys(clear=True)

        with open('test_log.log', 'r') as f:
            content = f.read()

        self.assertNotIn('supersecret123', content)
        self.assertIn('***MASKED***', content)

    def test_invalid_log_mode_raises_error(self):
        """Invalid logMode should raise ValueError with helpful message."""
        with self.assertRaises(ValueError) as context:
            ic.configureOutput(logMode='append')

        self.assertIn("logMode must be 'a' (append) or 'w' (overwrite)", str(context.exception))
        self.assertIn("'append'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            ic.configureOutput(logMode='x')

        self.assertIn("logMode must be 'a' (append) or 'w' (overwrite)", str(context.exception))
        self.assertIn("'x'", str(context.exception))

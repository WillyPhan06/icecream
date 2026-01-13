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

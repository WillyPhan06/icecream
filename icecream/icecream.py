#!/usr/bin/env python
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

import ast
import enum
import inspect
import pprint
import re
import sys
from collections import deque
from types import FrameType
from typing import Deque, Optional, cast, Any, Callable, Generator, List, Sequence, Set, Tuple, Type, Union, cast, Literal
import warnings
from datetime import datetime
import functools
from contextlib import contextmanager
from os.path import basename, realpath
from textwrap import dedent

import colorama
import executing
from pygments import highlight

# See https://gist.github.com/XVilka/8346728 for color support in various
# terminals and thus whether to use Terminal256Formatter or
# TerminalTrueColorFormatter.
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer as PyLexer, Python3Lexer as Py3Lexer

from .coloring import SolarizedDark


class Sentinel(enum.Enum):
    absent = object()


def bindStaticVariable(name: str, value: Any) -> Callable:
    def decorator(fn: Callable) -> Callable:
        setattr(fn, name, value)
        return fn
    return decorator


@bindStaticVariable('formatter', Terminal256Formatter(style=SolarizedDark))
@bindStaticVariable('lexer', Py3Lexer(ensurenl=False))
def colorize(s: str) -> str:
    self = colorize
    return highlight(
        s,
        cast(Py3Lexer, self.lexer),
        cast(Terminal256Formatter, self.formatter)
    )  # pyright: ignore[reportFunctionMemberAccess]


@contextmanager
def supportTerminalColorsInWindows() -> Generator:
    # Filter and replace ANSI escape sequences on Windows with equivalent Win32
    # API calls. This code does nothing on non-Windows systems.
    if sys.platform.startswith('win'):
        colorama.init()
        yield
        colorama.deinit()
    else:
        yield


def stderrPrint(*args: object) -> None:
    print(*args, file=sys.stderr)


def isLiteral(s: str) -> bool:
    try:
        ast.literal_eval(s)
    except Exception:
        return False
    return True


def colorizedStderrPrint(s: str) -> None:
    colored = colorize(s)
    with supportTerminalColorsInWindows():
        stderrPrint(colored)


def colorizedStdoutPrint(s: str) -> None:
    colored = colorize(s)
    with supportTerminalColorsInWindows():
        print(colored)


def safe_pformat(obj: object, *args: Any, **kwargs: Any) -> str:
    """pprint.pformat() with a couple of small safety/usability tweaks.

    In addition to the usual TypeError handling below, we special–case
    "medium sized" flat lists. For those, the standard pprint heuristics
    sometimes choose a one-item-per-line layout which makes the order of
    values hard to visually follow in ic()'s output. For such lists we
    prefer the more compact repr()-style representation.
    """

    def _pformat(extra_kwargs: Optional[dict] = None) -> str:
        # Helper so we always pass the same args/kwargs to pprint.
        final_kwargs = dict(kwargs)
        if extra_kwargs:
            final_kwargs.update(extra_kwargs)
        return pprint.pformat(obj, *args, **final_kwargs)

    try:
        # For flat lists we try a slightly wider layout first. This keeps
        # simple medium-sized lists on a single line in the common case.
        is_flat_list = (
            isinstance(obj, list)
            and not args
            and 'width' not in kwargs
            and not any(isinstance(el, (list, tuple, dict, set)) for el in obj)
        )
        if is_flat_list:
            formatted = _pformat({'width': 120})
        else:
            formatted = _pformat(None)
    except TypeError as e:
        # Sorting likely tripped on symbolic/elementwise comparisons.
        warnings.warn(f"pprint failed ({e}); retrying without dict sorting")
        try:
            # Py 3.8+: disable sorting globally for all nested dicts.
            return _pformat({'sort_dicts': False})
        except TypeError:
            # Py < 3.8: last-ditch, always works.
            return repr(obj)

    # Heuristic: if pprint decided to break a flat, medium-sized list across
    # many lines, fall back to repr() which keeps the list visually compact
    # and easier to read in ic()'s prefix/value layout.
    if is_flat_list and isinstance(obj, list) and 13 <= len(obj) <= 35:
        lines = formatted.splitlines()
        if len(lines) > 10:
            one_line = repr(obj)
            if len(one_line) <= 120:
                return one_line

    return formatted


DEFAULT_PREFIX = 'ic| '
DEFAULT_LINE_WRAP_WIDTH = 70  # Characters.
DEFAULT_CONTEXT_DELIMITER = '- '
DEFAULT_OUTPUT_FUNCTION = colorizedStderrPrint
DEFAULT_ARG_TO_STRING_FUNCTION = safe_pformat
DEFAULT_INDENTATION_STR = '    '  # 4 spaces per level.
DEFAULT_SENSITIVE_PLACEHOLDER = '***MASKED***'

"""
This info message is printed instead of the arguments when icecream
fails to find or access source code that's required to parse and analyze.
This can happen, for example, when

  - ic() is invoked inside a REPL or interactive shell, e.g. from the
    command line (CLI) or with python -i.

  - The source code is mangled and/or packaged, e.g. with a project
    freezer like PyInstaller.

  - The underlying source code changed during execution. See
    https://stackoverflow.com/a/33175832.
"""
NO_SOURCE_AVAILABLE_WARNING_MESSAGE = (
    'Failed to access the underlying source code for analysis. Was ic() '
    'invoked in a REPL (e.g. from the command line), a frozen application '
    '(e.g. packaged with PyInstaller), or did the underlying source code '
    'change during execution?')


def callOrValue(obj: object) -> object:
    return obj() if callable(obj) else obj


def _getCallStackDepth() -> int:
    """Get the current call stack depth.

    Returns the number of frames in the call stack, excluding this function itself.
    """
    depth = 0
    frame = inspect.currentframe()
    if frame is not None:
        frame = frame.f_back  # Skip this function
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


class Source(executing.Source):
    def get_text_with_indentation(self, node: ast.expr) -> str:
        result = self.asttokens().get_text(node)
        if '\n' in result:
            result = ' ' * node.first_token.start[1] + result # type: ignore[attr-defined]
            result = dedent(result)
        result = result.strip()
        return result


def prefixLines(prefix: str, s: str, startAtLine: int = 0) -> List[str]:
    lines = s.splitlines()

    for i in range(startAtLine, len(lines)):
        lines[i] = prefix + lines[i]

    return lines


def prefixFirstLineIndentRemaining(prefix: str, s: str) -> List[str]:
    indent = ' ' * len(prefix)
    lines = prefixLines(indent, s, startAtLine=1)
    lines[0] = prefix + lines[0]
    return lines


def formatPair(prefix: str, arg: Union[str, Sentinel], value: str) -> str:
    if arg is Sentinel.absent:
        argLines = []
        valuePrefix = prefix
    else:
        argLines = prefixFirstLineIndentRemaining(prefix, arg)
        valuePrefix = argLines[-1] + ': '

    looksLikeAString = (value[0] + value[-1]) in ["''", '""']
    if looksLikeAString:  # Align the start of multiline strings.
        valueLines = prefixLines(' ', value, startAtLine=1)
        value = '\n'.join(valueLines)

    valueLines = prefixFirstLineIndentRemaining(valuePrefix, value)
    lines = argLines[:-1] + valueLines
    return '\n'.join(lines)


class _SingleDispatchCallable:
    def __call__(self, *args: object) -> str:
        # This is a marker class, not a real thing you should use
        raise NotImplemented
    
    register: Callable[[Type], Callable]


def singledispatch(func: Callable) -> _SingleDispatchCallable:
    func = functools.singledispatch(func)

    # add unregister based on https://stackoverflow.com/a/25951784
    assert func.register.__closure__ is not None
    closure = dict(zip(func.register.__code__.co_freevars,
                       func.register.__closure__))
    registry = closure['registry'].cell_contents
    dispatch_cache = closure['dispatch_cache'].cell_contents

    def unregister(cls: Type) -> None:
        del registry[cls]
        dispatch_cache.clear()

    func.unregister = unregister  # type: ignore[attr-defined]
    return cast(_SingleDispatchCallable, func)


@singledispatch
def argumentToString(obj: object) -> str:
    s = DEFAULT_ARG_TO_STRING_FUNCTION(obj)
    s = s.replace('\\n', '\n')  # Preserve string newlines in output.
    return s


@argumentToString.register(str)
def _(obj: str) -> str:
    if '\n' in obj:
        return "'''" + obj + "'''"

    return "'" + obj.replace('\\', '\\\\') + "'"


def is_sensitive_key(key: str, sensitive_keys: Set[str]) -> bool:
    """Check if a key name matches any sensitive key pattern.

    Matching is case-insensitive and uses substring matching, so
    'password' will match 'PASSWORD', 'user_password', 'passwordHash', etc.

    Args:
        key: The key name to check.
        sensitive_keys: Set of sensitive key patterns to match against.

    Returns:
        True if the key matches any sensitive pattern, False otherwise.
    """
    if not sensitive_keys:
        return False
    key_lower = str(key).lower()
    sensitive_keys_lower = {k.lower() for k in sensitive_keys}
    for sensitive in sensitive_keys_lower:
        if sensitive in key_lower:
            return True
    return False


def _mask_sensitive_value(
    obj: object,
    sensitive_keys: Set[str],
    placeholder: str
) -> object:
    """Recursively mask sensitive values in objects.

    This function processes various data types and masks sensitive values:
    - For dicts: masks values where the key matches a sensitive pattern
    - For lists/tuples/sets: recursively processes each element
    - For strings: masks patterns like "password=xxx" or "api_key: xxx"

    The string pattern matching uses regex to find key-value patterns in the
    format: <sensitive_key><separator><value> where separator is '=' or ':'
    with optional whitespace.

    Args:
        obj: The object to mask sensitive values in.
        sensitive_keys: Set of key names (case-insensitive) to mask.
        placeholder: The placeholder string to use for masked values.

    Returns:
        A copy of the object with sensitive values masked.
    """
    if not sensitive_keys:
        return obj

    # Lowercase all sensitive keys for case-insensitive matching
    sensitive_keys_lower = {k.lower() for k in sensitive_keys}

    def mask_recursive(value: object) -> object:
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if is_sensitive_key(str(k), sensitive_keys):
                    result[k] = placeholder
                else:
                    result[k] = mask_recursive(v)
            return result
        elif isinstance(value, list):
            return [mask_recursive(item) for item in value]
        elif isinstance(value, tuple):
            return tuple(mask_recursive(item) for item in value)
        elif isinstance(value, set):
            return {mask_recursive(item) for item in value}
        elif isinstance(value, str):
            # Mask patterns like "password=xxx" or "api_key: xxx" in strings
            # Regex pattern explanation:
            #   - Group 1: (<sensitive_key>\s*[:=]\s*) - captures the key and separator
            #     - <sensitive_key> - the sensitive key name (escaped for regex)
            #     - \s* - optional whitespace
            #     - [:=] - either ':' or '=' as separator
            #     - \s* - optional whitespace after separator
            #   - Group 2: ([^\s,}}\]"\'`;]+) - captures the value to be masked
            #     - [^\s,}}\]"\'`;]+ - one or more characters that are NOT:
            #       - \s (whitespace)
            #       - , (comma - common delimiter)
            #       - }} (closing brace)
            #       - \] (closing bracket)
            #       - "\'` (quotes)
            #       - ; (semicolon - common delimiter)
            result = value
            for sensitive in sensitive_keys_lower:
                pattern = rf'({re.escape(sensitive)}\s*[:=]\s*)([^\s,}}\]"\'`;]+)'
                result = re.sub(pattern, rf'\1{placeholder}', result, flags=re.IGNORECASE)
            return result
        else:
            return value

    return mask_recursive(obj)


class IceCreamDebugger:
    _pairDelimiter = ', '  # Used by the tests in tests/.
    lineWrapWidth = DEFAULT_LINE_WRAP_WIDTH
    contextDelimiter = DEFAULT_CONTEXT_DELIMITER

    def __init__(self, prefix: Union[str, Callable[[], str]] =DEFAULT_PREFIX,
                 outputFunction: Callable[[str], None]=DEFAULT_OUTPUT_FUNCTION,
                 argToStringFunction: Union[_SingleDispatchCallable, Callable[[Any], str]]=argumentToString, includeContext: bool=False,
                 contextAbsPath: bool=False, enableIndentation: bool=False,
                 indentationStr: str=DEFAULT_INDENTATION_STR,
                 sensitiveKeys: Optional[Set[str]]=None,
                 sensitivePlaceholder: str=DEFAULT_SENSITIVE_PLACEHOLDER):
        self.enabled = True
        self.prefix = prefix
        self.includeContext = includeContext
        self.outputFunction = outputFunction
        self.argToStringFunction = argToStringFunction
        self.contextAbsPath = contextAbsPath
        self.enableIndentation = enableIndentation
        self.indentationStr = indentationStr
        self._baseStackDepth: Optional[int] = None
        self._sensitiveKeys: Set[str] = sensitiveKeys if sensitiveKeys is not None else set()
        self._sensitivePlaceholder = sensitivePlaceholder
        # Call limiting state
        self._callCount = 0
        self._outputCount = 0
        self._limitFirst: Optional[int] = None
        self._limitLast: Optional[int] = None
        self._limitEvery: Optional[int] = None
        # Using deque for O(1) popleft operations when buffer exceeds limitLast
        self._lastNBuffer: Deque[Tuple[str, int]] = deque()  # (output, indentLevel) tuples

    def __call__(self, *args: object) -> object:
        if self.enabled:
            # Increment call counter for every ic() call
            self._callCount += 1
            currentCallNum = self._callCount

            currentFrame = inspect.currentframe()
            assert currentFrame is not None and currentFrame.f_back is not None
            callFrame = currentFrame.f_back

            # Calculate indentation level based on call stack depth
            indentLevel = 0
            if self.enableIndentation:
                # +2 accounts for __call__ and _getCallStackDepth frames
                currentDepth = _getCallStackDepth() + 2
                if self._baseStackDepth is None:
                    self._baseStackDepth = currentDepth
                indentLevel = max(0, currentDepth - self._baseStackDepth)

            out = self._format(callFrame, *args)
            if indentLevel > 0:
                indent = self.indentationStr * indentLevel
                out = '\n'.join(indent + line for line in out.splitlines())

            # Determine if this call should produce output
            shouldOutput = self._shouldOutput(currentCallNum)

            # Handle limitLast buffering
            if self._limitLast is not None:
                # Buffer this output for later flushing
                self._lastNBuffer.append((out, indentLevel))
                # Keep only the last N entries (O(1) operation with deque)
                if len(self._lastNBuffer) > self._limitLast:
                    self._lastNBuffer.popleft()
            elif shouldOutput:
                self.outputFunction(out)
                self._outputCount += 1

        if not args:  # E.g. ic().
            passthrough = None
        elif len(args) == 1:  # E.g. ic(1).
            passthrough = args[0]
        else:  # E.g. ic(1, 2, 3).
            passthrough = args

        return passthrough

    def _shouldOutput(self, callNum: int) -> bool:
        """Determine if the current call should produce output based on limits.

        This method checks limitFirst and limitEvery settings to decide if
        output should be produced. limitLast is handled separately via buffering.

        Args:
            callNum: The current call number (1-indexed).

        Returns:
            True if output should be produced, False otherwise.
        """
        # If no limits are set, always output
        if self._limitFirst is None and self._limitEvery is None:
            return True

        shouldOutput = True

        # Check limitFirst: only output if within first N calls
        if self._limitFirst is not None:
            if callNum > self._limitFirst:
                shouldOutput = False

        # Check limitEvery: only output every Nth call (1st, N+1th, 2N+1th, ...)
        # When combined with limitFirst, both conditions must be satisfied
        if self._limitEvery is not None and shouldOutput:
            # Call numbers are 1-indexed, so call 1 should output,
            # then call 1+N, 1+2N, etc.
            if (callNum - 1) % self._limitEvery != 0:
                shouldOutput = False

        return shouldOutput

    def format(self, *args: object) -> str:
        currentFrame = inspect.currentframe()
        assert currentFrame is not None and currentFrame.f_back is not None
        callFrame = currentFrame.f_back
        out = self._format(callFrame, *args)
        return out

    def _format(self, callFrame: FrameType, *args: object) -> str:
        prefix = cast(str, callOrValue(self.prefix))

        context = self._formatContext(callFrame)
        if not args:
            time = self._formatTime()
            out = prefix + context + time
        else:
            if not self.includeContext:
                context = ''
            out = self._formatArgs(
                callFrame, prefix, context, args)

        return out

    def _formatArgs(self, callFrame: FrameType, prefix: str, context: str, args: Sequence[object]) -> str:
        callNode = Source.executing(callFrame).node
        if callNode is not None:
            assert isinstance(callNode, ast.Call)
            source = cast(Source, Source.for_frame(callFrame))
            sanitizedArgStrs = [
                source.get_text_with_indentation(arg)
                for arg in callNode.args]
        else:
            warnings.warn(
                NO_SOURCE_AVAILABLE_WARNING_MESSAGE,
                category=RuntimeWarning, stacklevel=4)
            sanitizedArgStrs = [Sentinel.absent] * len(args)

        pairs = list(zip(sanitizedArgStrs, cast(List[str], args)))

        out = self._constructArgumentOutput(prefix, context, pairs)
        return out

    def _maskAndConvert(self, val: object) -> str:
        """Mask sensitive values and convert to string representation.

        This method first masks any sensitive values in the object using
        the configured sensitive keys and placeholder, then converts the
        result to a string using argToStringFunction.

        Args:
            val: The value to mask and convert.

        Returns:
            String representation with sensitive values masked.
        """
        masked = _mask_sensitive_value(
            val, self._sensitiveKeys, self._sensitivePlaceholder)
        return self.argToStringFunction(masked)

    def _constructArgumentOutput(self, prefix: str, context: str, pairs: Sequence[Tuple[Union[str, Sentinel], str]]) -> str:
        def argPrefix(arg: str) -> str:
            return '%s: ' % arg

        pairs = [(arg, self._maskAndConvert(val)) for arg, val in pairs]
        # For cleaner output, if <arg> is a literal, eg 3, "a string",
        # b'bytes', etc, only output the value, not the argument and the
        # value, because the argument and the value will be identical or
        # nigh identical. Ex: with ic("hello"), just output
        #
        #   ic| 'hello',
        #
        # instead of
        #
        #   ic| "hello": 'hello'.
        #
        # When the source for an arg is missing we also only print the value,
        # since we can't know anything about the argument itself.
        pairStrs = [
            val if (arg is Sentinel.absent or isLiteral(arg))
            else (argPrefix(arg) + val)
            for arg, val in pairs]

        allArgsOnOneLine = self._pairDelimiter.join(pairStrs)
        multilineArgs = len(allArgsOnOneLine.splitlines()) > 1

        contextDelimiter = self.contextDelimiter if context else ''
        allPairs = prefix + context + contextDelimiter + allArgsOnOneLine
        firstLineTooLong = len(allPairs.splitlines()[0]) > self.lineWrapWidth

        if multilineArgs or firstLineTooLong:
            # ic| foo.py:11 in foo()
            #     multilineStr: 'line1
            #                    line2'
            #
            # ic| foo.py:11 in foo()
            #     a: 11111111111111111111
            #     b: 22222222222222222222
            if context:
                lines = [prefix + context] + [
                    formatPair(len(prefix) * ' ', arg, value)
                    for arg, value in pairs
                ]
            # ic| multilineStr: 'line1
            #                    line2'
            #
            # ic| a: 11111111111111111111
            #     b: 22222222222222222222
            else:
                argLines = [
                    formatPair('', arg, value)
                    for arg, value in pairs
                ]
                lines = prefixFirstLineIndentRemaining(prefix, '\n'.join(argLines))
        # ic| foo.py:11 in foo()- a: 1, b: 2
        # ic| a: 1, b: 2, c: 3
        else:
            lines = [prefix + context + contextDelimiter + allArgsOnOneLine]

        return '\n'.join(lines)

    def _formatContext(self, callFrame: FrameType) -> str:
        filename, lineNumber, parentFunction = self._getContext(callFrame)

        if parentFunction != '<module>':
            parentFunction = '%s()' % parentFunction

        context = '%s:%s in %s' % (filename, lineNumber, parentFunction)
        return context

    def _formatTime(self) -> str:
        now = datetime.now()
        formatted = now.strftime('%H:%M:%S.%f')[:-3]
        return ' at %s' % formatted

    def _getContext(self, callFrame: FrameType) -> Tuple[str, int, str]:
        frameInfo = inspect.getframeinfo(callFrame)
        lineNumber = frameInfo.lineno
        parentFunction = frameInfo.function

        filepath = (realpath if self.contextAbsPath else basename)(frameInfo.filename)  # type: ignore[operator]
        return filepath, lineNumber, parentFunction

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def use_stdout(self) -> None:
        self.outputFunction = colorizedStdoutPrint

    def use_stderr(self) -> None:
        self.outputFunction = colorizedStderrPrint

    def resetIndentation(self) -> None:
        """Reset the base stack depth for indentation.

        Call this method when you want the next ic() call to be treated
        as the new baseline (zero indentation level). This is useful when
        starting a new debugging session or when the call context has changed.
        """
        self._baseStackDepth = None

    def resetCallLimit(self) -> None:
        """Reset call limit counters and clear the lastN buffer.

        Call this method when starting a new debugging session to reset the
        call counter to zero and clear any buffered output from limitLast.
        The limit settings (limitFirst, limitLast, limitEvery) are preserved.
        """
        self._callCount = 0
        self._outputCount = 0
        self._lastNBuffer.clear()

    def flushCallLimit(self) -> None:
        """Flush the buffered output from limitLast mode.

        When limitLast is configured, ic() calls are buffered instead of being
        printed immediately. Call this method to print all buffered output
        (the last N calls). After flushing, the buffer is cleared.

        If limitLast is not configured, this method does nothing.
        """
        if self._limitLast is not None and self._lastNBuffer:
            for out, indentLevel in self._lastNBuffer:
                self.outputFunction(out)
                self._outputCount += 1
            self._lastNBuffer.clear()

    def configureOutput(
        self: "IceCreamDebugger",
        prefix: Union[str, Literal[Sentinel.absent]] = Sentinel.absent,
        outputFunction: Union[Callable, Literal[Sentinel.absent]] = Sentinel.absent,
        argToStringFunction: Union[Callable, Literal[Sentinel.absent]] = Sentinel.absent,
        includeContext: Union[bool, Literal[Sentinel.absent]] = Sentinel.absent,
        contextAbsPath: Union[bool, Literal[Sentinel.absent]] = Sentinel.absent,
        lineWrapWidth: Union[bool, Literal[Sentinel.absent]] = Sentinel.absent,
        enableIndentation: Union[bool, Literal[Sentinel.absent]] = Sentinel.absent,
        indentationStr: Union[str, Literal[Sentinel.absent]] = Sentinel.absent,
        limitFirst: Union[Optional[int], Literal[Sentinel.absent]] = Sentinel.absent,
        limitLast: Union[Optional[int], Literal[Sentinel.absent]] = Sentinel.absent,
        limitEvery: Union[Optional[int], Literal[Sentinel.absent]] = Sentinel.absent
    ) -> None:
        """Configure ic() output settings.

        Args:
            prefix: String or callable that returns a string to prefix output.
                Default is 'ic| '.
            outputFunction: Function to handle output. Receives the formatted
                string as argument. Default prints to stderr with syntax highlighting.
            argToStringFunction: Function to convert argument values to strings.
                Default uses pprint.pformat().
            includeContext: If True, include filename, line number, and function
                name in output. Default is False.
            contextAbsPath: If True, use absolute paths in context. Only applies
                when includeContext is True. Default is False.
            lineWrapWidth: Maximum line width before wrapping. Default is 70.
            enableIndentation: If True, ic() output is indented based on call
                stack depth relative to the first ic() call. This helps visualize
                nested and recursive function calls. The first ic() call establishes
                the baseline (zero indentation), and subsequent calls from deeper
                in the call stack are indented proportionally. Toggling this setting
                on/off preserves the baseline; use resetIndentation() to reset it.
                Default is False.
            indentationStr: String used for each level of indentation when
                enableIndentation is True. Default is 4 spaces ('    ').
            limitFirst: If set to a positive integer N, only the first N ic()
                calls will produce output. Subsequent calls are counted but
                suppressed. Set to None to disable. Default is None.

                Use case: When debugging loops or recursive functions, you often
                want to see the initial behavior without being overwhelmed by
                thousands of outputs. For example, when debugging a loop that
                processes 10,000 items, setting limitFirst=10 lets you verify
                the first few iterations work correctly before the output
                becomes unmanageable. This is also useful for recursive functions
                where you want to track entry into the recursion but don't need
                to see every single call.

            limitLast: If set to a positive integer N, only the last N ic()
                calls will produce output. All calls are buffered, and only
                the last N are printed when flushCallLimit() is called.
                Set to None to disable. Default is None.

                Use case: When you need to understand how a long-running process
                ended up in a particular state, the final calls are often most
                relevant. For example, when debugging why a recursive algorithm
                returned an unexpected result, seeing the last 10 ic() calls
                shows you the final steps that led to that result. This is
                particularly useful for debugging convergence issues, final
                state problems, or understanding the exit path of complex
                algorithms. Remember to call flushCallLimit() after your code
                runs to see the buffered output.

            limitEvery: If set to a positive integer N, only every Nth ic()
                call will produce output (1st, N+1th, 2N+1th, etc.).
                Set to None to disable. Default is None.

                Use case: When you want to see the "big picture" of how a loop
                or recursive function progresses without reading every single
                call. For example, when debugging a loop that processes 1,000
                items, setting limitEvery=100 shows you calls 1, 101, 201, etc.,
                giving you 10 evenly-spaced snapshots of the execution. This
                helps identify trends, patterns, or the point where behavior
                changes without wading through excessive output.

        Note:
            - limitFirst and limitEvery can be combined: output appears for
              calls that satisfy BOTH conditions. For example, limitFirst=50
              with limitEvery=10 shows calls 1, 11, 21, 31, 41 (every 10th
              call within the first 50).
            - limitLast operates independently and buffers output until
              flushCallLimit() is called.
            - Use resetCallLimit() to reset call counters and clear buffers
              when starting a new debugging session.

        Raises:
            TypeError: If no arguments are provided.

        Example:
            # Enable indentation for nested calls
            ic.configureOutput(enableIndentation=True)

            def inner():
                ic("inner")  # Indented output

            def outer():
                ic("outer")  # Baseline (no indentation)
                inner()

            outer()
            # Output:
            # ic| 'outer'
            #     ic| 'inner'

            # Custom indentation string
            ic.configureOutput(indentationStr='>> ')

            # Reset baseline for new debugging session
            ic.resetIndentation()

            # Debug a loop: see only first 5 iterations to verify initial behavior
            ic.configureOutput(limitFirst=5)
            ic.resetCallLimit()
            for i in range(1000):
                ic(i)  # Only shows i=0,1,2,3,4

            # Debug a loop: see every 100th call to track overall progress
            ic.configureOutput(limitFirst=None, limitEvery=100)
            ic.resetCallLimit()
            for i in range(1000):
                ic(i)  # Shows i=0,100,200,...,900

            # Debug to see final state: buffer all calls, show last 10
            ic.configureOutput(limitEvery=None, limitLast=10)
            ic.resetCallLimit()
            for i in range(1000):
                ic(i)  # All buffered
            ic.flushCallLimit()  # Prints i=990,991,...,999

            # Reset counters for new debugging session
            ic.resetCallLimit()
        """
        noParameterProvided = all(
            v is Sentinel.absent for k, v in locals().items() if k != 'self')
        if noParameterProvided:
            raise TypeError('configureOutput() missing at least one argument')

        if prefix is not Sentinel.absent:
            self.prefix = prefix

        if outputFunction is not Sentinel.absent:
            self.outputFunction = outputFunction

        if argToStringFunction is not Sentinel.absent:
            self.argToStringFunction = argToStringFunction

        if includeContext is not Sentinel.absent:
            self.includeContext = includeContext

        if contextAbsPath is not Sentinel.absent:
            self.contextAbsPath = contextAbsPath

        if lineWrapWidth is not Sentinel.absent:
            self.lineWrapWidth = lineWrapWidth

        if enableIndentation is not Sentinel.absent:
            self.enableIndentation = enableIndentation

        if indentationStr is not Sentinel.absent:
            self.indentationStr = indentationStr

        if limitFirst is not Sentinel.absent:
            self._limitFirst = limitFirst

        if limitLast is not Sentinel.absent:
            self._limitLast = limitLast
            # Clear the buffer when limitLast changes
            self._lastNBuffer.clear()

        if limitEvery is not Sentinel.absent:
            self._limitEvery = limitEvery

    def configureSensitiveKeys(
        self,
        keys: Optional[Sequence[str]] = None,
        placeholder: Union[str, Literal[Sentinel.absent]] = Sentinel.absent,
        add: Optional[Sequence[str]] = None,
        remove: Optional[Sequence[str]] = None,
        clear: bool = False
    ) -> None:
        """Configure sensitive key masking for ic() output.

        This method allows you to specify keys (like 'password', 'api_key',
        'token', etc.) whose values should be masked in ic() output to prevent
        accidental exposure of sensitive information.

        Key matching is case-insensitive and uses substring matching, so
        configuring 'password' will mask keys like 'password', 'PASSWORD',
        'user_password', 'passwordHash', etc.

        Args:
            keys: A sequence of key names to set as sensitive. This replaces
                any existing sensitive keys. Use None to leave existing keys
                unchanged.
            placeholder: The string to display instead of sensitive values.
                Default is '***MASKED***'.
            add: A sequence of key names to add to the existing sensitive keys.
            remove: A sequence of key names to remove from the sensitive keys.
            clear: If True, clear all sensitive keys before applying other
                operations. Default is False.

        Raises:
            TypeError: If no arguments are provided.

        Example:
            # Basic usage - mask password and api_key
            ic.configureSensitiveKeys(keys=['password', 'api_key', 'token'])

            password = 'secret123'
            ic(password)  # Output: ic| password: '***MASKED***'

            config = {'api_key': 'sk-abc123', 'debug': True}
            ic(config)  # Output: ic| config: {'api_key': '***MASKED***', 'debug': True}

            # Add more keys to existing configuration
            ic.configureSensitiveKeys(add=['secret', 'auth'])

            # Remove a key from masking
            ic.configureSensitiveKeys(remove=['token'])

            # Custom placeholder
            ic.configureSensitiveKeys(placeholder='[REDACTED]')

            # Clear all sensitive keys (disable masking)
            ic.configureSensitiveKeys(clear=True)
        """
        noParameterProvided = (
            keys is None and
            placeholder is Sentinel.absent and
            add is None and
            remove is None and
            not clear
        )
        if noParameterProvided:
            raise TypeError('configureSensitiveKeys() missing at least one argument')

        if clear:
            self._sensitiveKeys = set()

        if keys is not None:
            self._sensitiveKeys = set(keys)

        if add is not None:
            self._sensitiveKeys.update(add)

        if remove is not None:
            self._sensitiveKeys -= set(remove)

        if placeholder is not Sentinel.absent:
            self._sensitivePlaceholder = placeholder

    @property
    def sensitiveKeys(self) -> Set[str]:
        """Get the current set of sensitive keys."""
        return self._sensitiveKeys.copy()

    @property
    def sensitivePlaceholder(self) -> str:
        """Get the current placeholder string for masked values."""
        return self._sensitivePlaceholder

    @property
    def callCount(self) -> int:
        """Get the current call count since last reset."""
        return self._callCount

    @property
    def outputCount(self) -> int:
        """Get the number of calls that produced output since last reset."""
        return self._outputCount

    @property
    def limitFirst(self) -> Optional[int]:
        """Get the current limitFirst setting."""
        return self._limitFirst

    @property
    def limitLast(self) -> Optional[int]:
        """Get the current limitLast setting."""
        return self._limitLast

    @property
    def limitEvery(self) -> Optional[int]:
        """Get the current limitEvery setting."""
        return self._limitEvery


ic = IceCreamDebugger()

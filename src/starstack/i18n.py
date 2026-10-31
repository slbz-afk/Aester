"""Presentation-only localization. Never translate protocol keys or recipes."""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import os
from pathlib import Path
import re


def normalize_language(value) -> str:
    code = str(value or '').strip().replace('-', '_').lower()
    return 'en' if code == 'en' or code.startswith('en_') else 'zh_CN'


_language = normalize_language(os.environ.get('AESTER_LANGUAGE'))


def set_language(language: str) -> str:
    global _language
    _language = normalize_language(language)
    return _language


def get_language() -> str:
    return _language


def startup_language(settings=None) -> str:
    """Preserve a user's choice; otherwise follow this install's language.

The environment is the explicit stdio host/test override when no preference
exists. Reading the installer hint cannot change settings or processing data.
"""
    if settings is not None:
        saved = settings.value('appearance/language', None)
        if saved:
            return normalize_language(saved)
    explicit = os.environ.get('AESTER_LANGUAGE')
    if explicit:
        return normalize_language(explicit)
    root = os.environ.get('AESTER_INSTALL_ROOT')
    if root:
        try:
            with (Path(root) / 'install-language.txt').open(encoding='utf-8-sig') as stream:
                return normalize_language(stream.read(64))
        except (OSError, UnicodeError):
            pass
    return 'zh_CN'


_REGISTERED: list[str] = []                         # front-end catalogues, in registration order
_BUILTIN = ('starstack.translations_onboarding',)  # the engine's own exact catalogue
_MODULE_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\Z')


def register_catalogs(*modules: str) -> None:
    """A front end (the desktop app) adds its English catalogues by module name.

    Lookup order is the registered modules (in order) then the built-in one; later catalogues win on equal
    keys, the same gui -> dialogs -> onboarding order as before the open/closed split. Registering again is
    harmless; a missing registered module is skipped when the catalogue is built.
    """
    for name in modules:
        if not isinstance(name, str) or not _MODULE_NAME.match(name):
            raise ValueError('invalid catalogue module name')
        if name not in _REGISTERED:
            _REGISTERED.append(name)
    english_catalog.cache_clear()
    _translate_english.cache_clear()


@lru_cache(maxsize=1)
def english_catalog() -> dict[str, str]:
    catalog = {}
    for module in (*_REGISTERED, *_BUILTIN):
        try:
            entries = import_module(module).EN
        except ModuleNotFoundError as exc:
            if exc.name != module:
                raise
            continue
        catalog.update(entries)
    return catalog


def tr(source: str, *args, **kwargs) -> str:
    text = english_catalog().get(source, source) if get_language() == 'en' else source
    return text.format(*args, **kwargs) if args or kwargs else text


# --------------------------------------------------------------------------
# Backend messages. Exceptions and record fields (message/error/reason/...)
# stay Chinese in the core and on disk; tests and recovery logic depend on
# them. ``translate_message`` renders them in English at display time only.

_HAN = re.compile('[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
_ENVELOPE = re.compile(
    r'\A(\w+(?:Error|Exception|Cancelled|Busy|Damaged|Conflict|Denied|Required|Interrupt)): (.*)\Z', re.S)
_OS_ERROR = re.compile(r'\A\[(WinError|Errno) (-?\d+)\] (.*?)(: .*)?\Z', re.S)
# A localized Windows message inside longer text ends with an ideographic full stop.
_INLINE_OS_ERROR = re.compile('\\[(WinError|Errno) (-?\\d+)\\] ([^\\[\\]\\n]*?\u3002)')
# ``{name}`` any text, ``{name:d}`` a number, ``!m`` translate the capture too.
_TEMPLATE_TOKEN = re.compile(r'\{\{|\}\}|\{([A-Za-z_]\w*)(:d)?(!m)?\}')
_NUMBER = r'[-+]?\d+(?:\.\d+)?'
_SENTENCE_END = '\u3002\uff1b\uff01\uff1f'          # full stop, semicolon, exclamation, question
_SENTENCE_PUNCTUATION = dict(zip(_SENTENCE_END, '.;!?'))
_OPENING = '(\uff08[\u3010'
_CLOSING = ')\uff09]\u3011'
_COLON = '\uff1a'
_ENUMERATION = '\u3001'
# Chinese punctuation joining data that needs no translation (frame names, numbers).
_FIELD_PUNCTUATION = {'\u3001': ', ', '\uff0c': ', ', '\uff1b': '; ', '\uff1a': ': ',
                      '\uff08': ' (', '\uff09': ')', '\u3002': '. '}
_MAX_DEPTH = 12
_MAX_SENTENCE_RUN = 8            # composite templates span a few sentences, not whole reports
_MAX_TEMPLATE_TEXT = 8000        # longer text (log dumps) is only split, never regex-scanned whole

# English wording of common OS errors whose text Windows localizes. Python
# strips the trailing full stop of English system messages; so do we.
WINDOWS_ERRORS = {
    1: 'Incorrect function',
    2: 'The system cannot find the file specified',
    3: 'The system cannot find the path specified',
    5: 'Access is denied',
    6: 'The handle is invalid',
    8: 'Not enough memory resources are available to process this command',
    13: 'The data is invalid',
    15: 'The system cannot find the drive specified',
    17: 'The system cannot move the file to a different disk drive',
    19: 'The media is write protected',
    21: 'The device is not ready',
    23: 'Data error (cyclic redundancy check)',
    32: 'The process cannot access the file because it is being used by another process',
    33: 'The process cannot access the file because another process has locked a portion of the file',
    39: 'The disk is full',
    50: 'The request is not supported',
    53: 'The network path was not found',
    67: 'The network name cannot be found',
    80: 'The file exists',
    87: 'The parameter is incorrect',
    109: 'The pipe has been ended',
    112: 'There is not enough space on the disk',
    123: 'The filename, directory name, or volume label syntax is incorrect',
    145: 'The directory is not empty',
    183: 'Cannot create a file when that file already exists',
    193: 'Not a valid Win32 application',
    206: 'The filename or extension is too long',
    225: 'Operation did not complete successfully because the file contains a virus or potentially '
         'unwanted software',
    232: 'The pipe is being closed',
    267: 'The directory name is invalid',
    740: 'The requested operation requires elevation',
    995: 'The I/O operation has been aborted because of either a thread exit or an application request',
    1117: 'The request could not be performed because of an I/O device error',
    1224: 'The requested operation cannot be performed on a file with a user-mapped section open',
    1314: 'A required privilege is not held by the client',
    1392: 'The file or directory is corrupted and unreadable',
    1450: 'Insufficient system resources exist to complete the requested service',
    1455: 'The paging file is too small for this operation to complete',
    1920: 'The file cannot be accessed by the system',
    10054: 'An existing connection was forcibly closed by the remote host',
    10060: 'The connection attempt timed out because the other side did not respond',
    10061: 'No connection could be made because the target machine actively refused it',
}
ERRNO_ERRORS = {
    1: 'Operation not permitted',
    2: 'No such file or directory',
    5: 'Input/output error',
    9: 'Bad file descriptor',
    12: 'Not enough memory',
    13: 'Permission denied',
    16: 'Device or resource busy',
    17: 'File exists',
    18: 'Invalid cross-device link',
    20: 'Not a directory',
    21: 'Is a directory',
    22: 'Invalid argument',
    24: 'Too many open files',
    28: 'No space left on device',
    30: 'Read-only file system',
    32: 'Broken pipe',
    38: 'File name too long',
    41: 'Directory not empty',
}


def _parse_template(text: str) -> list:
    """``str`` literals and ``(name, recursive, numeric)`` fields; ``{{``/``}}`` are literal braces."""
    parts: list = []
    literal: list[str] = []
    position = 0
    for match in _TEMPLATE_TOKEN.finditer(text):
        literal.append(text[position:match.start()])
        token = match.group(0)
        if token in ('{{', '}}'):
            literal.append(token[0])
        else:
            parts.append(''.join(literal))
            literal = []
            parts.append((match.group(1), bool(match.group(3)), bool(match.group(2))))
        position = match.end()
    literal.append(text[position:])
    parts.append(''.join(literal))
    return [part for part in parts if part != '']


def _message_catalog():
    try:
        from . import translations_messages as catalog
    except ModuleNotFoundError as exc:
        if exc.name != __name__.rpartition('.')[0] + '.translations_messages':
            raise
        return {}, []
    return catalog.EN_EXACT, catalog.EN_PATTERNS


def _capture(field: tuple, rest: list) -> str:
    """Regex for one field. A text field stops at the first occurrence of the literal
    that follows it, so matching stays linear (no backtracking across literals)."""
    if field[2]:
        body = _NUMBER
    elif rest and isinstance(rest[0], str):
        body = f'(?:(?!{re.escape(rest[0])}).)*?'
    else:
        body = '.*?'
    return f'(?P<{field[0]}>{body})'


@lru_cache(maxsize=1)
def message_patterns() -> tuple:
    """Compiled ``(anchor, regex, recursive_fields, english_parts)``, most specific first.

    ``anchor`` is the template's longest literal, a cheap prefilter before the regex.
    English parts are ``str`` literals or 1-tuples naming a captured field.
    """
    compiled = []
    for chinese, english in _message_catalog()[1]:
        source = _parse_template(chinese)
        regex = ''.join(re.escape(part) if isinstance(part, str) else _capture(part, source[index + 1:])
                        for index, part in enumerate(source))
        recursive = {part[0] for part in source if not isinstance(part, str) and part[1]}
        target = _parse_template(english)
        recursive |= {part[0] for part in target if not isinstance(part, str) and part[1]}
        literals = [part for part in source if isinstance(part, str)]
        specificity = sum(map(len, literals))
        compiled.append((specificity, max(literals, key=len, default=''), re.compile(r'\A' + regex + r'\Z', re.S),
                         frozenset(recursive),
                         tuple(part if isinstance(part, str) else (part[0],) for part in target)))
    compiled.sort(key=lambda item: -item[0])
    return tuple(item[1:] for item in compiled)


def _exact(text: str) -> str | None:
    # The GUI catalogues win, so tr() and translate_message() never disagree.
    found = english_catalog().get(text)
    return found if found is not None else _message_catalog()[0].get(text)


def _os_error(code: int, kind: str) -> str | None:
    return (WINDOWS_ERRORS if kind == 'WinError' else ERRNO_ERRORS).get(code)


def _render(target: tuple, match, recursive: frozenset, depth: int) -> str:
    # Captured text is inserted literally: paths may contain braces or '%'.
    out = []
    for part in target:
        if isinstance(part, str):
            out.append(part)
        else:
            value = match.group(part[0])
            out.append(_translate_field(value, depth) if part[0] in recursive else value)
    return ''.join(out)


def _translate_field(value: str, depth: int) -> str:
    """A captured sub-message. Chinese punctuation joining data (frame names, numbers)
    becomes ASCII; anything that looks like a path is left exactly as it is."""
    if _HAN.search(value):
        return _translate(value, depth + 1)
    if '\\' in value or '/' in value or not any(mark in value for mark in _FIELD_PUNCTUATION):
        return value
    for mark, replacement in _FIELD_PUNCTUATION.items():
        value = value.replace(mark, replacement)
    return re.sub(' {2,}', ' ', value).strip()


def _direct(core: str, depth: int) -> str | None:
    """Whole-text rules: catalogue entry, exception envelope, template, OS error."""
    found = _exact(core)
    if found is not None:
        return found
    envelope = _ENVELOPE.match(core)
    if envelope and _HAN.search(envelope.group(2)):
        return envelope.group(1) + ': ' + _translate(envelope.group(2), depth + 1)
    if len(core) <= _MAX_TEMPLATE_TEXT:          # bounded regex work on huge log dumps
        for anchor, regex, recursive, target in message_patterns():
            match = regex.match(core) if anchor in core else None
            if match:
                return _render(target, match, recursive, depth)
    system = _OS_ERROR.match(core)
    if system and _HAN.search(system.group(3)):
        english = _os_error(int(system.group(2)), system.group(1))
        if english is not None:
            return f'[{system.group(1)} {system.group(2)}] {english}{system.group(4) or ""}'
    return None


def _split_sentences(text: str) -> list[str]:
    """Split after sentence punctuation that is not inside brackets."""
    pieces, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char in _OPENING:
            depth += 1
        elif char in _CLOSING:
            depth = max(0, depth - 1)
        elif char in _SENTENCE_END and depth == 0:
            pieces.append(text[start:index + 1])
            start = index + 1
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def _translate_sentences(text: str, depth: int) -> str | None:
    pieces = _split_sentences(text)
    if len(pieces) < 2:
        return None
    rendered: list[tuple[str, bool]] = []
    index = 0
    while index < len(pieces):
        # Prefer the longest run of sentences that one entry renders completely.
        for end in range(min(len(pieces), index + _MAX_SENTENCE_RUN), index + 1, -1):
            if index == 0 and end == len(pieces):
                continue                      # the caller already tried the whole text
            found = _direct(''.join(pieces[index:end]), depth + 1)
            if found is not None and not _HAN.search(found):
                rendered.append((found, True))
                index = end
                break
        else:
            piece = pieces[index]
            result = _translate(piece, depth + 1)
            if result == piece and piece[-1] in _SENTENCE_END:
                bare = _translate(piece[:-1], depth + 1)
                if bare != piece[:-1]:
                    result = bare if bare.endswith(('.', ';', '!', '?')) else bare + _SENTENCE_PUNCTUATION[piece[-1]]
            rendered.append((result, result != piece))
            index += 1
    if not any(changed for _, changed in rendered):
        return None
    out = rendered[0][0]
    for (_, changed_before), (current, changed) in zip(rendered, rendered[1:]):
        joiner = ' ' if (changed_before or changed) and not out.endswith((' ', '\n')) else ''
        out += joiner + current
    return out


def _translate_colon_prefix(text: str, depth: int) -> str | None:
    """A known Chinese head before the first full-width colon, then any detail."""
    head, colon, tail = text.partition(_COLON)
    if not colon or not _HAN.search(head):
        return None
    prefix = english_catalog().get(head + _COLON)
    if prefix is None:
        found = _translate(head, depth + 1)
        if _HAN.search(found):
            return None
        prefix = found.rstrip().rstrip(':') + ': '
    return prefix + _translate_field(tail, depth).lstrip()


def _translate_enumeration(text: str, depth: int) -> str | None:
    """An enumeration whose items all have an English rendering."""
    if _ENUMERATION not in text:
        return None
    items = [_translate(item, depth + 1) for item in text.split(_ENUMERATION)]
    return None if any(_HAN.search(item) for item in items) else ', '.join(items)


def _translate(text: str, depth: int = 0) -> str:
    if depth > _MAX_DEPTH or not _HAN.search(text):
        return text
    core = text.strip()
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    found = _direct(core, depth)
    if found is None and '\n' in core:
        found = '\n'.join(_translate(line, depth + 1) for line in core.split('\n'))
    if found is None:
        found = _translate_colon_prefix(core, depth)
    if found is None:
        found = _translate_sentences(core, depth)
    if found is None:
        found = _translate_enumeration(core, depth)
    if found is None:
        def replace(match):
            english = _os_error(int(match.group(2)), match.group(1))
            return match.group(0) if english is None else f'[{match.group(1)} {match.group(2)}] {english}'
        found = _INLINE_OS_ERROR.sub(replace, core)
    return lead + found + trail


@lru_cache(maxsize=4096)
def _translate_english(text: str) -> str:
    return _translate(text)


def translate_message(text, language: str | None = None) -> str:
    """Render a backend message for display; the stored text never changes.

    Chinese UI and messages without Chinese pass through unchanged. English UI
    tries, in order: an exact catalogue entry, ``TypeError: ...`` envelopes
    (type kept), anchored templates from ``translations_messages`` (captured
    text is inserted literally, never through ``str.format``), localized
    ``[WinError N]``/``[Errno N]`` system messages, line-by-line text, a known
    head before a full-width colon, sentence-by-sentence text and enumerations.
    Anything unknown is shown as it is; ``None`` becomes an empty string.
    """
    if text is None:
        return ''
    text = str(text)
    effective = get_language() if language is None else normalize_language(language)
    if effective != 'en' or not _HAN.search(text):
        return text
    return _translate_english(text)

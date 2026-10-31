"""English presentation of backend messages (display-time only).

Backend exceptions and record fields stay Chinese; many tests pin them. The
GUI shows them through ``i18n.translate_message``. This file scans every
module the GUI and its worker processes import, collects the Chinese message
shapes they can emit, and requires an English rendering for each one. Items
not covered yet live in a shrink-only allowlist.
"""
from __future__ import annotations

import ast
import itertools
import json
import re
import string
from pathlib import Path

import pytest

from starstack import i18n, translations_messages
from starstack.i18n import translate_message

_SRC = Path(__file__).resolve().parents[2] / 'src'
# The GUI is the GPL aester_desktop package; the backend messages it shows come from the Apache-2.0 starstack
# engine. Module names are unique across the two packages, so a name is looked up in either source folder.
SOURCES = (_SRC / 'aester_desktop', _SRC / 'starstack')
ALLOWLIST = Path(__file__).resolve().parents[1] / 'data' / 'untranslated_messages_allowlist.json'


def _source(name: str) -> Path:
    for folder in SOURCES:
        if (folder / f'{name}.py').is_file():
            return folder / f'{name}.py'
    raise FileNotFoundError(name)


def _exists(name: str) -> bool:
    return any((folder / f'{name}.py').is_file() for folder in SOURCES)
ROOTS = ('gui', 'worker', 'transient_scan_worker')
FIELDS = frozenset({'message', 'error', 'reason', 'notice', 'detail', 'note', 'label'})
# Functions whose returned text is raised or stored as a message.
MESSAGE_FUNCTION_WORDS = ('message', 'refusal', 'reason', 'notice')
# Translation catalogues and user guides are presentation text, not backend messages.
EXCLUDED = frozenset({'i18n', 'translations_gui', 'translations_dialogs', 'translations_onboarding',
                      'translations_messages', 'user_guide', 'user_guide_en'})
HAN = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')
# Han plus CJK/fullwidth punctuation: a clean English rendering has none of these.
CJK = re.compile(r'[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]')
SAMPLE = '7'
_WRAPPERS = {'str', 'repr', 'int', 'float', 'len', 'round', 'abs', 'Path', 'sorted', 'list', 'tuple', 'set',
             'format', 'tr', 'bool', 'min', 'max', 'sum', 'fmt'}


# --------------------------------------------------------------------------- scanner

def _imports(name: str) -> set[str]:
    tree = ast.parse(_source(name).read_text(encoding='utf-8'))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {alias.name.split('.')[1] for alias in node.names
                      if alias.name.startswith(('starstack.', 'aester_desktop.'))}
        elif isinstance(node, ast.ImportFrom):
            if node.level == 1:
                found |= ({alias.name for alias in node.names} if node.module is None
                          else {node.module.split('.')[0]})
            elif node.module in ('starstack', 'aester_desktop'):
                found |= {alias.name for alias in node.names}
            elif node.module and node.module.startswith(('starstack.', 'aester_desktop.')):
                found.add(node.module.split('.')[1])
    return {module for module in found if _exists(module)}


def reachable_modules() -> list[str]:
    """Modules (desktop and engine) imported, transitively, by the GUI and the worker processes."""
    seen: set[str] = set()
    stack = list(ROOTS)
    while stack:
        name = stack.pop()
        if name not in seen:
            seen.add(name)
            stack.extend(_imports(name) - seen)
    return sorted(seen - EXCLUDED)


class Slot:
    """A dynamic part of a message: a data value, or a set of literal choices."""

    def __init__(self, name: str, choices: list[str] | None = None):
        self.name = name
        self.choices = choices


def _slot_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) \
                and node.slice.value.isidentifier():
            return node.slice.value
        return _slot_name(node.value)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in _WRAPPERS and node.args:
            return _slot_name(node.args[0])
        if isinstance(func, ast.Attribute) and func.attr in {'join', 'get', 'format'} and node.args:
            return _slot_name(node.args[0])
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
        return node.value
    if isinstance(node, ast.BinOp):
        return _slot_name(node.left)
    if isinstance(node, ast.IfExp):
        return 'choice'
    if isinstance(node, ast.FormattedValue):
        return _slot_name(node.value)
    return 'value'


def _choices(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        body, orelse = _choices(node.body), _choices(node.orelse)
        return None if body is None or orelse is None else body + orelse
    return None


def _slot(node: ast.AST) -> Slot:
    return Slot(_slot_name(node), _choices(node))


def _parts(node: ast.AST) -> list:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [value.value if isinstance(value, ast.Constant) else _slot(value.value) for value in node.values]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'format' \
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str):
        parts, index = [], 0
        for literal, field, _, _ in string.Formatter().parse(node.func.value.value):
            parts.append(literal)
            if field is not None:
                if field and not field.isdigit():
                    parts.append(Slot(field.split('.')[0].split('[')[0]))
                else:
                    arg = node.args[int(field) if field else index] if node.args else None
                    parts.append(_slot(arg) if arg is not None else Slot('value'))
                    index += 1
        return parts
    return [_slot(node)]


def message_shapes(node: ast.AST | None) -> list[list]:
    """Every message shape an expression can produce, as literal text and slots."""
    if node is None:
        return []
    if isinstance(node, ast.IfExp):
        return message_shapes(node.body) + message_shapes(node.orelse)
    if isinstance(node, ast.BoolOp):
        return [shape for value in node.values for shape in message_shapes(value)]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tr':
        return []                     # already localized at the call site by the GUI catalogues
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'str' and node.args:
        return message_shapes(node.args[0])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get' \
            and len(node.args) == 2:
        return message_shapes(node.args[1])          # record.get('message', '默认文字')
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = message_shapes(node.left) or [_parts(node.left)]
        right = message_shapes(node.right) or [_parts(node.right)]
        return [a + b for a in left for b in right]
    if isinstance(node, ast.Constant):
        return [[node.value]] if isinstance(node.value, str) else []
    if isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'format'
            and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)):
        return [_parts(node)]
    return []


def shape_key(parts: list) -> str:
    """Chinese template with ``{}`` for each dynamic part (literal braces doubled)."""
    return ''.join(part.replace('{', '{{').replace('}', '}}') if isinstance(part, str) else '{}'
                   for part in parts)


def shape_samples(parts: list, limit: int = 12) -> list[str]:
    choices = [[part] if isinstance(part, str) else (part.choices or [SAMPLE]) for part in parts]
    return [''.join(combo) for combo in itertools.islice(itertools.product(*choices), limit)]


class _Collector(ast.NodeVisitor):
    def __init__(self, module: str):
        self.module = module
        self.items: list[tuple[str, int, str, list]] = []
        self.functions: list[str] = []

    def add(self, node: ast.AST, kind: str, value: ast.AST | None) -> None:
        for parts in message_shapes(value):
            if any(isinstance(part, str) and HAN.search(part) for part in parts):
                self.items.append((self.module, node.lineno, kind, parts))

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Raise(self, node):
        if isinstance(node.exc, ast.Call):
            for arg in node.exc.args:
                self._raised(node, arg)
        self.generic_visit(node)

    def _raised(self, node, arg):
        if message_shapes(arg):
            self.add(node, 'raise', arg)
        elif isinstance(arg, ast.Call):      # raise X(helper(log, f'...'))
            for inner in [*arg.args, *(keyword.value for keyword in arg.keywords)]:
                self._raised(node, inner)

    def visit_Call(self, node):
        for keyword in node.keywords:
            if keyword.arg in FIELDS:
                self.add(node, 'keyword', keyword.value)
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == '__init__' and isinstance(func.value, ast.Call) \
                and isinstance(func.value.func, ast.Name) and func.value.func.id == 'super':
            for arg in node.args:
                self.add(node, 'exception', arg)
        self.generic_visit(node)

    def _assigned(self, node, targets, value):
        for target in targets:
            if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) \
                    and target.slice.value in FIELDS:
                self.add(node, 'field', value)
            elif isinstance(target, ast.Name) and target.id in FIELDS:
                self.add(node, 'variable', value)

    def visit_Assign(self, node):
        self._assigned(node, node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._assigned(node, [node.target], node.value)
        self.generic_visit(node)

    def visit_Dict(self, node):
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value in FIELDS:
                self.add(node, 'field', value)
        self.generic_visit(node)

    def visit_Return(self, node):
        if self.functions and any(word in self.functions[-1] for word in MESSAGE_FUNCTION_WORDS):
            self.add(node, 'return', node.value)
        self.generic_visit(node)


def scan_messages(modules: list[str] | None = None) -> dict[str, dict]:
    """``shape_key -> {'parts', 'sites'}`` for every Chinese message shape."""
    shapes: dict[str, dict] = {}
    for module in modules or reachable_modules():
        collector = _Collector(module)
        collector.visit(ast.parse(_source(module).read_text(encoding='utf-8')))
        for module_name, line, kind, parts in collector.items:
            entry = shapes.setdefault(shape_key(parts), {'parts': parts, 'sites': []})
            entry['sites'].append(f'{module_name}:{line}:{kind}')
    return shapes


def _untranslated(parts: list) -> list[str]:
    return [sample for sample in shape_samples(parts) if CJK.search(translate_message(sample, 'en'))]


def _allowlist() -> dict[str, list[str]]:
    data = json.loads(ALLOWLIST.read_text(encoding='utf-8'))
    return {item['template']: item['sites'] for item in data['untranslated']}


# --------------------------------------------------------------------------- tests

@pytest.fixture(scope='module')
def shapes():
    return scan_messages()


@pytest.fixture(autouse=True)
def restore_language():
    previous = i18n.get_language()
    yield
    i18n.set_language(previous)


def test_scanner_reaches_worker_modules_and_known_messages(shapes):
    modules = set(reachable_modules())
    assert {'workflow', 'core', 'siril', 'references', 'storage', 'regions', 'editing_project',
            'trail_review_session', 'workflow_edit', 'star_density', 'milkyway_gate', 'components',
            'preview', 'region_preview', 'raw_audit', 'orientation', 'transient_scan'} <= modules
    assert not modules & EXCLUDED
    assert len(shapes) > 800
    for message in ('用户请求停止', '已停止，可从完整阶段恢复', '用户取消', '尚未找到 Siril 1.4.x，请选择 siril-cli.exe'):
        assert message in shapes
    assert any(key.startswith('Siril {} 超过{}时限') for key in shapes)          # exception __init__
    assert any(key.startswith('背景检测到{}个紧致星点候选') for key in shapes)    # f-string + detail
    assert any(key.startswith('{}{}；转换日志：') for key in shapes)              # *_message return


def test_every_scanned_backend_message_has_english(shapes):
    allowed = _allowlist()
    missing = {key: entry['sites'][:2] for key, entry in shapes.items()
               if key not in allowed and _untranslated(entry['parts'])}
    assert not missing, 'Add English to translations_messages.py for:\n' + '\n'.join(
        f'{sites} {key}' for key, sites in sorted(missing.items()))


def test_allowlist_only_lists_real_untranslated_messages(shapes):
    """Shrink-only: stale or newly translated entries must be removed from the JSON."""
    data = json.loads(ALLOWLIST.read_text(encoding='utf-8'))
    templates = [item['template'] for item in data['untranslated']]
    assert templates == sorted(set(templates))
    assert data['count'] == len(templates)
    for template in templates:
        assert template in shapes, f'no longer emitted; remove from allowlist: {template}'
        assert _untranslated(shapes[template]['parts']), f'now translated; remove from allowlist: {template}'


def test_catalogue_entries_are_well_formed():
    for chinese, english in translations_messages.EN_EXACT.items():
        assert HAN.search(chinese), chinese
        assert english.strip() and not CJK.search(english), (chinese, english)
    seen = set()
    for chinese, english in translations_messages.EN_PATTERNS:
        assert chinese not in seen, chinese
        seen.add(chinese)
        source, target = i18n._parse_template(chinese), i18n._parse_template(english)
        source_fields = [part[0] for part in source if not isinstance(part, str)]
        target_fields = [part[0] for part in target if not isinstance(part, str)]
        # Placeholder parity: every capture is shown exactly once, nothing is invented.
        assert sorted(source_fields) == sorted(target_fields), (chinese, english)
        assert len(set(source_fields)) == len(source_fields), chinese
        assert any(isinstance(part, str) and HAN.search(part) for part in source), chinese
        assert not CJK.search(english), english
        # Adjacent captures are ambiguous; templates keep a literal between them.
        assert all(isinstance(a, str) or isinstance(b, str) for a, b in zip(source, source[1:])), chinese


def test_each_pattern_renders_its_own_english():
    """No other entry shadows a pattern; captures are inserted verbatim."""
    for chinese, english in translations_messages.EN_PATTERNS:
        values = {part[0]: ('7' if part[2] else f'<{part[0]}>')
                  for part in i18n._parse_template(chinese) if not isinstance(part, str)}
        sample = ''.join(part if isinstance(part, str) else values[part[0]]
                         for part in i18n._parse_template(chinese))
        expected = ''.join(part if isinstance(part, str) else values[part[0]]
                           for part in i18n._parse_template(english))
        assert translate_message(sample, 'en') == expected, chinese


def test_chinese_ui_and_messages_without_chinese_are_unchanged():
    for text in ('用户请求停止', 'RuntimeError: 堆栈作业未成功：x', '[WinError 5] 拒绝访问。'):
        assert translate_message(text, 'zh_CN') == text
    i18n.set_language('zh_CN')
    assert translate_message('用户请求停止') == '用户请求停止'
    i18n.set_language('en')
    assert translate_message('用户请求停止') == 'Stop requested by user'
    assert translate_message('已停止，可从完整阶段恢复') == 'Stopped. You can resume from the last complete stage.'
    assert translate_message(None) == ''
    for text in ('', 'plain {0} text %s', "[WinError 5] Access is denied: 'C:\\x'"):
        assert translate_message(text, 'en') == text
    assert translate_message('完全未知的消息。另一句未知。', 'en') == '完全未知的消息。另一句未知。'


def test_captured_text_is_inserted_literally():
    path = r'C:\星空{night}\%s\{0}{{x}}.NEF'
    assert translate_message(f'素材不存在或格式不支持：{path}', 'en') == f'Source file missing or unsupported: {path}'
    assert translate_message(f'目标目录已存在：{path}', 'en') == f'Target folder already exists: {path}'
    assert (translate_message(f'亮迹扫描进程失败（3）；日志：{path}\nline {{1}}', 'en')
            == f'The trail scan process failed (3); log: {path}\nline {{1}}')
    # Full-width punctuation inside a captured path is never normalized.
    assert (translate_message('无法改名旧作业目录（D:\\jobs（1）\\a、b）；请关闭正在使用它的程序后重试', 'en')
            == 'Could not rename the old job folder (D:\\jobs（1）\\a、b). Close any program using it and try again.')


def test_nested_exception_envelopes_keep_their_type():
    assert (translate_message('RuntimeError: 堆栈作业未成功：ValueError: Siril 版本已变化，请创建新作业', 'en')
            == 'RuntimeError: Stacking did not finish: ValueError: The Siril version has changed. Please create a new job.')
    assert (translate_message('堆栈完成，但 M+D 导出未成功：RuntimeError: 颜色参考原片在读取期间发生变化', 'en')
            == 'Stacking finished, but the M+D export did not succeed: RuntimeError: '
               'The color reference original changed while being read')
    assert translate_message('WorkflowCancelled: 用户请求停止', 'en') == 'WorkflowCancelled: Stop requested by user'
    assert translate_message('KeyError: 未收录的内容', 'en') == 'KeyError: 未收录的内容'


def test_localized_os_errors_keep_code_and_path():
    assert (translate_message("[WinError 2] 系统找不到指定的文件。: 'C:\\\\a{1}.NEF'", 'en')
            == "[WinError 2] The system cannot find the file specified: 'C:\\\\a{1}.NEF'")
    assert translate_message('[WinError 5] 拒绝访问。', 'en') == '[WinError 5] Access is denied'
    assert (translate_message("PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。: 'a' -> 'b'", 'en')
            == "PermissionError: [WinError 32] The process cannot access the file because it is being used by "
               "another process: 'a' -> 'b'")
    assert (translate_message('银河增强失败：[WinError 112] 磁盘空间不足。', 'en')
            == 'Milky Way enhancement failed: [WinError 112] There is not enough space on the disk')
    assert (translate_message("无法改名旧作业目录（[WinError 183] 当文件已存在时，无法创建该文件。: 'D:\\\\j'）；"
                              "请关闭正在使用它的程序后重试", 'en')
            == "Could not rename the old job folder ([WinError 183] Cannot create a file when that file already "
               "exists: 'D:\\\\j'). Close any program using it and try again.")
    assert translate_message('[Errno 13] 权限不够', 'en') == '[Errno 13] Permission denied'
    assert translate_message('[WinError 99999] 未知错误。', 'en') == '[WinError 99999] 未知错误。'


def test_lookalike_text_does_not_backtrack_catastrophically():
    import time
    lookalike = '7 帧 × 单帧 8（9×10：转换帧 1 + 配准帧 2）= 3，' * 60 + '母版'
    unknown = '这是一段未收录的中文说明，包含逗号与数字 12，用于测试。' * 300
    started = time.perf_counter()
    for text in (lookalike, unknown, '未知' * 20000, '：'.join(['未知头'] * 300)):
        i18n._translate_english.cache_clear()
        assert translate_message(text, 'en') == text
    assert time.perf_counter() - started < 5


def test_multi_line_messages_translate_line_by_line():
    assert (translate_message('Siril convert 失败（1）\nlog: unsupported\n', 'en')
            == 'Siril convert failed (1)\nlog: unsupported\n')
    assert (translate_message('用户请求停止\n未知的一行\n已停止，可从完整阶段恢复', 'en')
            == 'Stop requested by user\n未知的一行\nStopped. You can resume from the last complete stage.')


def test_composite_worker_messages():
    done = ('处理完成：天空堆栈实际使用 14／16 帧，未参与 2 帧。未参与的是像质门槛排除的 a、b'
            '（a：stars 0.40x median；b：fwhm 1.90x median，wfwhm 1.70x median）。'
            '亮迹候选 3 条（建议标签：流星 2、未知 1），1 条具备加回条件并另出含亮迹成片；'
            '旧摘要尚未核实自动省略与未测量候选；请读取亮迹列表重新计算，当前不能保证没有遗漏。'
            '标签与策略可用 starstack transients 修改。线性母版、后期母版与成片已保存')
    english = translate_message(done, 'en')
    assert not CJK.search(english), english
    assert 'The sky stack used 14/16 frames; 2 frames were left out.' in english
    assert ('excluded by the image-quality gate: a, b (a: stars 0.40x median; '
            'b: fwhm 1.90x median, wfwhm 1.70x median).') in english
    assert 'Trail candidates: 3 (suggested labels: meteor 2, unknown 1).' in english
    assert english.endswith('Linear master, editing master and photograph saved.')
    conversion = ('当前 Siril／LibRaw 引擎无法解码此 RAW 文件的格式或压缩方式（日志报告 Unsupported file format or not RAW file）。'
                  '请保留原片，尝试用兼容转换器导出无额外有损压缩的 DNG 副本后重试。Siril convert 失败（1）\ntail；'
                  '转换日志：C:\\j\\convert.log（暂时无法读取）')
    english = translate_message('RuntimeError: ' + conversion, 'en')
    assert not CJK.search(english), english
    assert english.endswith('Siril convert failed (1)\ntail; conversion log: C:\\j\\convert.log (currently unreadable)')
    advice = ('背景检测到120个紧致星点候选，按显眼程度加权约350单位/百万参考像素；建议背景55%。'
              '银河密度样本不足，保守建议比背景低15个百分点。')
    assert not CJK.search(translate_message(advice, 'en'))
    timeout = ('Siril stack 超过总运行时限 600 秒，已请求停止；日志：C:\\j\\stack.log。'
               '静默计算也可能触发日志时限，可按素材调整时限后另行重试。')
    assert translate_message(timeout, 'en').startswith('Siril stack exceeded its total run time limit of 600 seconds')

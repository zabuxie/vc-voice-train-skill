# -*- coding: utf-8 -*-
"""
补齐 RVC 包缺失的 i18n 模块（幂等）

问题：
    RVC 20240604 整合包缺少 `i18n/` 目录，但 `infer/lib/train/process_ckpt.py`
    会 `from i18n.i18n import I18nAuto`，缺了就会报：
        ModuleNotFoundError: No module named 'i18n'

    影响：`-sw 1` 自动导出推理权重时失败，或手动转格式时失败。

修复：
    在 RVC 根目录写一个最小实现的 i18n 包（不覆盖已有实现）。

用法：
    <RVC>\\runtime\\python.exe fix_i18n.py [RVC根目录]
"""
import os
import sys

I18N_PY = '''# -*- coding: utf-8 -*-
"""Minimal I18nAuto compatible with upstream RVC.

Looks for locale files under <pkg>/i18n/<language>/. If none are found,
the call returns the original key, which is the upstream behaviour for
the Chinese locale where keys are already Chinese text.

DSH fix: RVC 20240604 bundle ships without this module, but
infer/lib/train/process_ckpt.py imports it.
"""
import json
import locale
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))


class I18nAuto:
    def __init__(self, language=None):
        if language is None:
            try:
                language = locale.getdefaultlocale()[0]
            except Exception:
                language = None
        self.language = language or "en"
        self.language_map = {}
        self._load()

    def _load(self):
        candidates = [self.language]
        if "_" in self.language:
            candidates.append(self.language.replace("_", "-"))
        for lang in candidates:
            d = os.path.join(_ROOT, lang)
            if not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(d, name), "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if isinstance(data, dict):
                    self.language_map.update(data)

    def __call__(self, key):
        if not isinstance(key, str):
            return key
        return self.language_map.get(key, key)

    def print(self, key):
        print(self(key))
'''

INIT_PY = ''


def find_rvc(start):
    p = os.path.abspath(start)
    for _ in range(8):
        if os.path.isfile(os.path.join(p, 'infer', 'modules', 'vc', 'pipeline.py')):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.environ.get('DSH_RVC_DIR', '')


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    rvc = find_rvc(start)
    if not rvc:
        print('[错误] 找不到 RVC 根目录。请在第一个参数传，或设 DSH_RVC_DIR')
        return 1

    print('=' * 64)
    print('RVC i18n 模块补齐')
    print('RVC 根:', rvc)
    print('=' * 64)

    d = os.path.join(rvc, 'i18n')
    init_f = os.path.join(d, '__init__.py')
    i18n_f = os.path.join(d, 'i18n.py')

    if os.path.isfile(init_f) and os.path.isfile(i18n_f):
        print('[跳过] i18n 模块已存在')
        # 验证可导入
        try:
            sys.path.insert(0, rvc)
            from i18n.i18n import I18nAuto  # noqa
            _ = I18nAuto('zh_CN')
            print('[验证] I18nAuto 可正常实例化')
        except Exception as e:
            print('[警告] 已存在但导入失败:', e)
        return 0

    os.makedirs(d, exist_ok=True)
    if not os.path.isfile(init_f):
        with open(init_f, 'w', encoding='utf-8', newline='\n') as f:
            f.write('')
        print('[写入]', init_f)
    if not os.path.isfile(i18n_f):
        with open(i18n_f, 'w', encoding='utf-8', newline='\n') as f:
            f.write(I18N_PY)
        print('[写入]', i18n_f)

    # 验证
    try:
        sys.path.insert(0, rvc)
        for m in list(sys.modules):
            if m.startswith('i18n'):
                del sys.modules[m]
        from i18n.i18n import I18nAuto
        inst = I18nAuto('zh_CN')
        print('[验证] I18nAuto 可正常实例化，示例输出:', inst('训练'))
    except Exception as e:
        print('[错误] 写入后导入失败:', e)
        return 2

    print()
    print('[成功] i18n 模块已补齐')
    return 0


if __name__ == '__main__':
    sys.exit(main())

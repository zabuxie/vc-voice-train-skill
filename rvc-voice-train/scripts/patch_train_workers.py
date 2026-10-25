# -*- coding: utf-8 -*-
"""
给 RVC 的 train.py 打 DataLoader workers=0 支持补丁（幂等，自动备份）

解决什么问题：
    Windows 上训练可能报 `WinError 1455 页面文件太小`。
    原因是 DataLoader 的 spawn 子进程重复加载 cuBLAS DLL，虚拟内存超限。
    把 DSH_NUM_WORKERS 设为 0 可让数据在主进程内加载，绕开 spawn。

为什么需要补丁：
    DataLoader 在 num_workers=0 时要求 prefetch_factor=None 且
    persistent_workers=False。原版 train.py 无条件传 int，会抛
    ValueError: prefetch_factor option could only be specified in multiprocessing

用法：
    <RVC>/runtime/python.exe patch_train_workers.py [RVC根目录]

不打这个补丁也可以：只要不用 DSH_NUM_WORKERS=0（用默认 1），但页面文件要够大。
"""
import os
import sys
import shutil
import datetime

MARKER = '_dsh_workers'

ORIGINAL = """    train_loader = DataLoader(
        train_dataset,
        num_workers=int(os.environ.get("DSH_NUM_WORKERS", "1")),
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=bool(int(os.environ.get("DSH_PERSISTENT", "0"))),
        prefetch_factor=int(os.environ.get("DSH_PREFETCH", "2")),
    )"""

PATCHED = """    # DSH patch: workers=0 时在主进程内加载数据（不 spawn 子进程）。
    # WinError 1455（页面文件太小）出现在 spawn 子进程重复加载 cuBLAS DLL 时，
    # 设 DSH_NUM_WORKERS=0 可绕开；此时 DataLoader 要求 prefetch_factor=None，
    # persistent_workers 也必须为 False。
    _dsh_workers = int(os.environ.get("DSH_NUM_WORKERS", "1"))
    if _dsh_workers == 0:
        _dsh_prefetch = None
        _dsh_persistent = False
    else:
        _dsh_prefetch = int(os.environ.get("DSH_PREFETCH", "2"))
        _dsh_persistent = bool(int(os.environ.get("DSH_PERSISTENT", "0")))
    train_loader = DataLoader(
        train_dataset,
        num_workers=_dsh_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=_dsh_persistent,
        prefetch_factor=_dsh_prefetch,
    )"""


def find_rvc(start):
    p = os.path.abspath(start)
    for _ in range(6):
        if os.path.isfile(os.path.join(p, 'infer', 'modules', 'train', 'train.py')):
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

    target = os.path.join(rvc, 'infer', 'modules', 'train', 'train.py')
    print('=' * 64)
    print('RVC train.py workers 补丁')
    print('目标:', target)
    print('=' * 64)

    if not os.path.isfile(target):
        print('[错误] train.py 不存在')
        return 1

    with open(target, 'r', encoding='utf-8', newline='') as f:
        src = f.read()

    if MARKER in src:
        print('[跳过] 补丁已存在（%d 处 %s）' % (src.count(MARKER), MARKER))
        return 0

    # 处理换行符差异
    if ORIGINAL not in src:
        crlf_orig = ORIGINAL.replace('\n', '\r\n')
        if crlf_orig in src:
            orig, patched = crlf_orig, PATCHED.replace('\n', '\r\n')
        else:
            print('[错误] 未找到预期的原版代码块')
            print('  请手动编辑: 查找 "prefetch_factor=int(os.environ.get"')
            print('  按本脚本里的 PATCHED 内容替换')
            return 2
    else:
        orig, patched = ORIGINAL, PATCHED

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    bak = target + '.bak_' + stamp
    shutil.copy2(target, bak)
    print('[备份]', os.path.basename(bak))

    new_src = src.replace(orig, patched, 1)
    with open(target, 'w', encoding='utf-8', newline='') as f:
        f.write(new_src)

    # 校验
    with open(target, 'r', encoding='utf-8', newline='') as f:
        check = f.read()
    if MARKER in check:
        try:
            import ast
            ast.parse(check)
            print('[成功] 补丁已应用，语法校验通过')
        except SyntaxError as e:
            print('[警告] 补丁已应用但语法校验失败:', e)
            print('  正在恢复备份')
            shutil.copy2(bak, target)
            return 3
        print()
        print('现在可以用:  export DSH_NUM_WORKERS=0')
        return 0
    else:
        print('[失败] 未检测到补丁标记，恢复备份')
        shutil.copy2(bak, target)
        return 3


if __name__ == '__main__':
    sys.exit(main())

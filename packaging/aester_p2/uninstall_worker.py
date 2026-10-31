"""Standalone uninstall entry: copied beside private bootstrap Python in TEMP."""
from pathlib import Path
import argparse
import json
import sys
import traceback

import install_lifecycle as lifecycle
import install_ownership as ownership
import uninstall_options as options


def format_inspection(installation, selection, language='zh_CN'):
    """Presentation only; never broaden the cleanup plan or its authority."""
    total = installation['owned_component_and_download_bytes']
    count = installation['owned_component_and_download_files']
    plans = selection.get('cache_plans', [])
    cache_bytes = sum(plan['bytes'] for plan in plans)
    preserved = len(selection.get('preserved', []))
    if language == 'en':
        return (
            'Aester and this installation\'s own runtime, Siril, nox and registered downloads will be removed.\n'
            'Original photographs, exported images, external backends and unknown files are kept. '
            'Shared components and older components without ownership records are also kept.\n\n'
            f'Owned components and downloads: {total / (1024 ** 3):.2f} GB in {count:,} files.\n\n'
            f'Optional cache cleanup: {len(plans)} completed jobs, {cache_bytes / (1024 ** 3):.2f} GB. '
            f'{preserved} unverified items will be kept.\n'
            'Only recorded, verified intermediate images with independently verified exports can be removed. '
            'Unfinished jobs, job records, unknown files and older unverified cache files stay in place.'
        )
    return (
        '将移除 Aester 程序及本安装独占的运行环境、Siril、nox 和已登记下载包。\n'
        '原片、独立保存的作品、外部后端和未知文件保留；共享组件与旧版无归属组件也会保留。\n\n'
        f'独占组件和下载包：{total / (1024 ** 3):.2f} GB，{count:,} 个文件。\n\n'
        f'可选缓存清理：{len(plans)} 个已完成作业，{cache_bytes / (1024 ** 3):.2f} GB；'
        f'{preserved} 项无法安全确认的内容将保留。\n'
        '仅清理有记录、校验一致且已有独立作品的图像中间产物。'
        '未完成作业、作业记录、未知文件及无法校验的旧缓存保留。'
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--result',required=True)
    parser.add_argument('--inspect',action='store_true')
    parser.add_argument('--clear-safe-cache',action='store_true')
    parser.add_argument('--clear-settings',action='store_true')
    parser.add_argument('--language',choices=('en','zh_CN'),default='zh_CN')
    args=parser.parse_args()
    root=ownership.safe(args.root)
    output=Path(args.result)
    result={'state':'failed','install_root':str(root),'errors':[]}
    try:
        if Path(__file__).resolve().is_relative_to(root):
            raise ValueError('卸载助手必须先复制到安装目录之外')
        ownership._identity(root)
        selection=options.scan_options(root)
        if args.inspect:
            result={'state':'ready','install_root':str(root),'options':selection}
            result['installation']=ownership.describe_installation(root)
            summary=format_inspection(result['installation'],selection,args.language)
            output.with_suffix('.txt').write_text(summary,encoding='utf-8-sig')
        else:
            # No payload mutation can precede the installation-wide gate.
            with lifecycle.exclusive_maintenance(root,mark_pending=True):
                result['cache_results']=[]
                if args.clear_safe_cache:
                    for plan in selection.get('cache_plans',[]):
                        outcome=options.execute_cache_cleanup(plan,exclusive=True)
                        result['cache_results'].append(outcome)
                        if outcome.get('errors'):raise RuntimeError('处理缓存清理未完成；保留卸载入口以供重试')
                if args.clear_settings:
                    result['settings_result']=options.execute_settings_cleanup(selection['settings_plan'],exclusive=True)
                    if result['settings_result'].get('errors'):raise RuntimeError('设置清理未完成；保留卸载入口以供重试')
                result['components']=ownership.cleanup(root)
                result['state']='complete'
                # Leave uninstall.pending in place while Inno removes code.
        ownership.atomic(output,result)
        return 0
    except Exception as error:
        result['errors'].append(str(error));result['traceback']=traceback.format_exc()
        ownership.atomic(output,result)
        summary = (
            'Uninstall did not finish. The uninstall entry is still available. '
            'Close Aester and related background jobs, resolve any file-access problem, then retry.\n\n'
            'Technical details: ' + str(output)
        ) if args.language == 'en' else (
            '卸载未完成。已保留卸载入口，请关闭 Aester 和相关后台任务，排除文件访问问题后重试。\n\n'
            '详细记录：' + str(output)
        )
        output.with_suffix('.txt').write_text(summary,encoding='utf-8-sig')
        print(str(error),file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())

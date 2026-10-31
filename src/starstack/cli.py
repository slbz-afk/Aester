import argparse
import json
import math
import time

from .core import doctor, submit_job, load_job, cancel_job, resume_job, create_demo, recompose_transients


def _finite_range(low: float, high: float):
    def parse(value: str) -> float:
        try:
            number = float(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("须为有限数值") from exc
        if not math.isfinite(number) or not low <= number <= high:
            raise argparse.ArgumentTypeError(f"须为 {low} 至 {high} 的有限数值")
        return number
    return parse


def _jpeg_quality(value: str) -> int:
    """--jpeg-quality: an integer from 60 to 100."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("须为 60 至 100 的整数") from exc
    if not 60 <= number <= 100:
        raise argparse.ArgumentTypeError("须为 60 至 100 的整数")
    return number


JPEG_QUALITY_HELP = "JPEG 质量 60–100（4:4:4 色度）；省略时保持原有编码"


def _quality_option(args) -> dict:
    return {"jpeg_quality": args.jpeg_quality} if args.jpeg_quality is not None else {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Starstack local prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--siril")
    demo = sub.add_parser("demo")
    demo.add_argument("--output", default="data/demo")
    run = sub.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--output", default="outputs")
    run.add_argument("--siril")
    run.add_argument("--mask")
    run.add_argument("--key")
    run.add_argument("--wait", action="store_true")
    run.add_argument("--support-mode", choices=("legacy", "strict_siril_1_4_4"), default="legacy",
                     help="显式试用完整插值支持候选（需天空蒙版与固定 Siril 构建）")
    run.add_argument("--stack-range-mode", choices=("legacy", "positive_power2_v1", "finite_validity_v1"), default="legacy",
                     help="保护堆栈高光；positive_power2_v1 单通道零值和负值照常堆栈（暗端在0处截断），遇非有限值停止并提示")
    run.add_argument("--frame-masks", help="逐帧天空蒙版目录（<原片文件名主干>.png，需同时给 --mask）")
    run.add_argument("--stack-engine", help="finite_validity_v1 使用的已验证浮点堆栈程序；配准仍用 --siril")
    run.add_argument("--transient-marks", help="亮迹搜索标记 JSON：[{input, x0, y0, x1, y1[, half_width_px, protect]}]；protect:true 显式请求保护，源帧原生坐标")
    run.add_argument("--force-restart", action="store_true",
                     help="同一 --key 的作业已结束（成功/失败/取消，含崩溃）时把旧目录改名为 job-<id>.restarted-<UTC> 保留并重跑；运行中或素材不同仍拒绝")
    sub.add_parser("help-guide", help="显示GUI与MCP共用指南")
    workflow = sub.add_parser("workflow", help="后台完整出图：堆栈、颜色、可选分层与缩星")
    workflow_sources = workflow.add_mutually_exclusive_group(required=True)
    workflow_sources.add_argument("--input", help="处理文件夹内的直接 RAW/FITS 文件")
    workflow_sources.add_argument("--files", nargs="+", help="只处理按此顺序明确选择的照片")
    workflow.add_argument("--output", default="outputs")
    workflow.add_argument("--siril")
    workflow.add_argument("--mask")
    workflow.add_argument("--key")
    workflow.add_argument("--stack-engine")
    workflow.add_argument("--stack-range-mode", choices=("legacy", "positive_power2_v1"), default="legacy")
    workflow.add_argument("--star-model")
    workflow.add_argument("--retain-trails", action="store_true", help="检测亮迹供复核；配合 --trail-default meteor 自动加回量测可靠的流星")
    workflow.add_argument("--trail-default", choices=("meteor", "review_only"),
                          help="meteor：基础成片完成后自动另存加回量测可靠流星的版本（算法建议）；省略或 review_only：只检测、不自动加回")
    workflow.add_argument("--reduce-stars", action="store_true", help="开启星点整理，须配置模型与天空蒙版")
    workflow.add_argument("--save-starless", action="store_true", help="另存无星图 starless_srgb16.tif（需 --reduce-stars）")
    workflow.add_argument("--save-star-layer", action="store_true",
                          help="另存星点加／减层 stars_add_srgb16.tif、stars_subtract_srgb16.tif（需 --reduce-stars）")
    workflow.add_argument("--warmth", type=_finite_range(-100, 100), default=0., help="相对冷暖；0沿用拍摄白平衡")
    workflow.add_argument("--tint", type=_finite_range(-100, 100), default=0., help="相对绿／洋红色调")
    workflow.add_argument("--exposure-ev", type=_finite_range(-4, 4), default=0., help="显示曝光补偿，不改写线性母版")
    workflow.add_argument("--background-strength", type=_finite_range(0, 1), help="省略时采用密度建议")
    workflow.add_argument("--adapter", help="模型适配参数JSON文件，不猜测权重的通道/范围")
    workflow.add_argument("--jpeg-quality", type=_jpeg_quality,
                          help="作品 JPEG 质量 60–100（4:4:4 色度，冻结在任务中，之后的亮迹版本沿用）；省略时保持原有编码")
    workflow.add_argument("--wait", action="store_true")
    for name in ("workflow-status", "workflow-cancel", "workflow-resume"):
        sub.add_parser(name).add_argument("workflow_dir")
    gc = sub.add_parser("gc", help="回收作业的被替代导出；默认只列清单，--apply 才删除；--work-cache 另删可由原片重建的转换帧/配准帧/序列文件/原片副本（之后不能恢复或重做亮迹）",
                        description="加固工作包 H3（docs/HARDENING_PACKAGE_20260922.md）：绝不触碰原片、母版、被引用的产物、记录与日志。")
    gc.add_argument("job")
    gc.add_argument("--apply", action="store_true", help="真正删除（默认 dry-run）")
    gc.add_argument("--work-cache", action="store_true", help="连同 work/light_*.fit、r_light*.fit、*.seq、stage/ 原片副本、range-*/trial*/ 一起（只允许已完成作业）")
    devices = sub.add_parser("devices", help="列出设备支持分级登记表（A 实测序列 / B+ 单帧核对 / B 声明支持 / C 已知问题）")
    devices.add_argument("--check", action="store_true", help="只做登记表自检")
    inspect_device = sub.add_parser("inspect-device", help="识别一张 RAW/DNG/FITS 的机型、查登记分级、探测元数据与颜色底子路线；非 DNG 的 RAW 经 LibRaw 读元数据时会解包 RAW 数据，不去马赛克、不改文件")
    inspect_device.add_argument("source")
    transients = sub.add_parser("transients", help="查看或修改已完成作业的亮迹候选处置（只重做亮迹层与含亮迹成片，不重跑堆栈）")
    transients.add_argument("job")
    transients.add_argument("--label", nargs=2, action="append", metavar=("ID", "CLASS"), help="人工标签：meteor/satellite/aircraft/artifact/unknown/auto")
    transients.add_argument("--override", nargs=2, action="append", metavar=("ID", "DECISION"), help="逐条处置：keep/omit/auto")
    transients.add_argument("--policy", action="append", metavar="CLASS=DECISION", help="策略，如 satellite=keep")
    transients.add_argument("--mark", nargs=5, action="append", metavar=("INPUT", "X0", "Y0", "X1", "Y1"), help="亮迹搜索走廊（源帧原生坐标）；配合 --protect 请求保护")
    transients.add_argument("--protect", action="store_true", help="对本次全部 --mark 显式请求加回；不改变分类，未能兑现会列出原因，override omit 可撤回")
    transients.add_argument("--note", nargs=2, action="append", metavar=("ID", "TEXT"))
    transients.add_argument("--full", action="store_true", help="打印完整候选记录")
    transients.add_argument("--needs-review", action="store_true", help="按优先级列出尚未核实的省略或未兑现保留请求；不改变处置")
    transients.add_argument("--layer-style", choices=("source_residual", "photographic_v3"), help="另存亮迹修订：原残差或抑制彩噪与硬端点的摄影派生")
    photo = sub.add_parser("finish-photo", help="实验全天空／星点／地景调整，摄影审美未通过",
                           description="实验全天空／星点／地景调整，摄影审美未通过；另存16位sRGB摄影派生。")
    photo.add_argument("--source", required=True)
    photo.add_argument("--output", default="outputs")
    photo.add_argument("--sky-mask", required=True)
    photo.add_argument("--protected-mask", required=True, help="需显式提供亮迹等保护区域，不自动猜测")
    photo.add_argument("--mode", choices=("photographic", "natural"), default="photographic")
    photo.add_argument("--detail-strength", type=float, default=1.)
    photo.add_argument("--star-strength", type=float, default=1.)
    photo.add_argument("--ground-strength", type=float, default=0.)
    photo.add_argument("--jpeg-quality", type=_jpeg_quality, help=JPEG_QUALITY_HELP)
    galaxy = sub.add_parser("finish-milkyway", help="通过显式银河渐变蒙版另存局部亮度、结构与颜色调整",
                            description="需同尺寸银河、天空与保护蒙版；只做局部展示，不是物理信号恢复。")
    galaxy.add_argument("--source", required=True)
    galaxy.add_argument("--output", default="outputs")
    galaxy.add_argument("--sky-mask", required=True)
    galaxy.add_argument("--protected-mask", required=True, help="亮迹保护范围；有配对背景时声明残差范围，否则冻结全部像素（可能形成暗带）")
    galaxy.add_argument("--milkyway-mask", required=True, help="同尺寸单通道银河权重蒙版，不自动定位")
    galaxy.add_argument("--milkyway-evidence", help="绑定源及银河蒙版SHA256的对象存在复核JSON；缺省或未确认时不增强")
    galaxy.add_argument("--transient-background", help="与源SHA256绑定的同尺寸16位sRGB背景；先增强背景再合回有符号亮迹残差")
    galaxy.add_argument("--starless-background", help="实验：绑定处理基底及天空/保护蒙版SHA256的16位sRGB去星背景；保留有符号星层，分离质量待验收")
    galaxy.add_argument("--brightness", type=float, default=.020)
    galaxy.add_argument("--contrast", type=float, default=.60)
    galaxy.add_argument("--structure", type=float, default=.90)
    galaxy.add_argument("--color", type=float, default=.35)
    galaxy.add_argument("--strength", type=float, default=.60)
    galaxy.add_argument("--jpeg-quality", type=_jpeg_quality, help=JPEG_QUALITY_HELP)
    starless = sub.add_parser("starless", help="实验：用 ONNX 去星模型（产品内 onnxruntime 分块推理）另存弥散估计与有符号星层；模型由用户自备，不随产品分发",
                              description="无星图工作包步 2（docs/STARLESS_PACKAGE_20260923.md §二）：显示域 16 位 sRGB 进、同尺寸弥散估计出，有符号残差由产品精确保留；锁定区（sky≠1 或保护）复制源。需要可选组件 onnxruntime。")
    starless.add_argument("--source", required=True, help="本产品输出的 16 位 sRGB TIFF")
    starless.add_argument("--sky-mask", required=True)
    starless.add_argument("--protected-mask", required=True)
    starless.add_argument("--model", required=True, help="ONNX 模型文件（用户自备，例如 tf2onnx 转换的 nox；不在仓库内）")
    starless.add_argument("--output", default="outputs", help="根目录，内建唯一新目录 separation-<hex12>")
    starless.add_argument("--tile", type=int, default=512)
    starless.add_argument("--stride", type=int, default=256, help="不大于 tile，且 tile − stride 为偶数")
    starless.add_argument("--stitch", choices=("blend", "crop"), default="blend", help="blend：Hann 加权拼接；crop：与 nox 同几何的中央裁切覆盖")
    starless.add_argument("--batch", type=int, default=1)
    starless.add_argument("--layout", choices=("nhwc", "nchw"), default="nhwc")
    starless.add_argument("--channel-order", choices=("rgb", "bgr"), default="rgb", help="喂给模型的通道序；复现 nox.exe 缓存用 bgr（它未做 BGR→RGB 转换），按 nox 训练语义用 rgb")
    starless.add_argument("--value-range", choices=("0..1", "-1..1"), default="0..1", help="模型图自身的取值范围；nox 请写 --value-range=-1..1（负范围用等号连接；x*2−1 进、(y+1)/2 出）")
    starless.add_argument("--provider", action="append", help="onnxruntime 执行提供者，可重复（缺省由 onnxruntime 选择）")
    starless.add_argument("--threads", type=int, help="onnxruntime intra-op 线程数（缺省 8，与工作流去星相同）")
    starless.add_argument("--transient-background", help="与源 SHA256 绑定的同尺寸亮迹背景（同 finish-milkyway）")
    layers = sub.add_parser("export-star-layers", help="导出可继续后期的背景、恒星与亮迹RGB16 TIFF图层包")
    layers.add_argument("--separation-result", required=True, help="已成功的当前分层result.json")
    layers.add_argument("--output", default="outputs")
    reduction = sub.add_parser("reduce-stars", help="从已有分层另存弱星整理或无星成片，保护独立亮迹",
                               description="非线性压低星层中低亮度部分；处理显示域亮度，不是测光星等调整或几何去卷积。新背景力度可从原样延伸至移除已分离星层，较高力度会渐进放开亮星保护；安全边界与亮迹余量仍受保护。")
    reduction.add_argument("--separation-result", required=True, help="已成功的当前分层result.json；使用其背景、星层及亮迹层")
    reduction.add_argument("--output", default="outputs")
    reduction.add_argument("--strength", type=float, default=.55, help="弱星压暗强度，0–1；0为原样合回")
    reduction.add_argument("--background-strength", type=_finite_range(0, 1), help="新全范围背景去星力度：0不移星，1移除安全范围内已分离的星层；省略沿用旧strength行为")
    reduction.add_argument("--milkyway-offset", type=_finite_range(-1, 1), default=-.2, help="银河相对背景的力度偏移−1至1，负值减弱；仅与background-strength及已确认银河依据共同生效")
    reduction.add_argument("--bright-start", type=float, default=.10, help="亮星保护过渡起点，显示域归一化亮度")
    reduction.add_argument("--bright-end", type=float, default=.45, help="亮星保护过渡终点，须大于起点")
    reduction.add_argument("--milkyway-mask", help="可选同尺寸银河渐变蒙版；需存在依据才放松银河内缩星")
    reduction.add_argument("--milkyway-evidence", help="绑定源及银河蒙版的存在复核JSON；不允许自动伪造人工确认")
    reduction.add_argument("--milkyway-relief", type=float, default=.5, help="已确认银河内的缩星减弱比例，0–1")
    reduction.add_argument("--bright-radius", type=int, help="亮星邻域保护半径0–32像素，0关闭扩展；缺省由核心确定")
    reduction.add_argument("--jpeg-quality", type=_jpeg_quality, help=JPEG_QUALITY_HELP)
    suggestion = sub.add_parser("suggest-stars", help="根据已有星层密度建议背景力度与银河偏移，只报告、不自动出图")
    suggestion.add_argument("--separation-result", required=True, help="已成功的当前分层result.json")
    suggestion.add_argument("--output", default="outputs")
    suggestion.add_argument("--milkyway-mask", help="可选同尺寸银河渐变蒙版")
    suggestion.add_argument("--milkyway-evidence", help="绑定源及银河蒙版的人工存在复核JSON；不能自动伪造确认")
    audit = sub.add_parser("inspect-raw", help="保存单张 RAW 的解码采样与数值检查，不执行显影")
    audit.add_argument("--source", required=True)
    audit.add_argument("--output", default="outputs")
    develop = sub.add_parser("develop-raw", help="实验：从 RAW 检查生成浮点双线性母版及固定颜色衍生")
    develop.add_argument("--audit", required=True)
    develop.add_argument("--output", default="outputs")
    develop.add_argument("--recipe", help="复用首帧的 color_recipe.json，不重新采用当前帧 WB")
    develop.add_argument("--exposure", type=float, default=2.0, help="仅预览曝光 EV；默认 +2")
    trace = sub.add_parser("protect-trace", help="从 RAW 检查记录保存人工亮迹区域及来源")
    trace.add_argument("--audit", required=True)
    trace.add_argument("--output", default="outputs")
    trace.add_argument("--rectangle", nargs=4, type=int, required=True, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    trace.add_argument("--classification", choices=("unknown", "meteor", "satellite", "aircraft", "artifact"), default="unknown")
    trace.add_argument("--note", default="")
    candidate = sub.add_parser("trace-status")
    candidate.add_argument("candidate")
    background = sub.add_parser('coordinate-background', help='从已完成堆栈另存背景协调派生与共同显示预览')
    background.add_argument('job')
    background.add_argument('--samples', help='全尺寸二值背景样本PNG，白色为明确背景')
    background.add_argument('--recipe', help='回放本作业已有成功配方，不重新拟合')
    background.add_argument('--mode', choices=('gradient', 'neutralize', 'both'))
    background.add_argument('--strength', type=float)
    background.add_argument('--red-gain', type=float)
    background.add_argument('--blue-gain', type=float)
    master = sub.add_parser("export-master", help="从已完成作业的相机 RGB 母版另存场景参照线性后期母版（M：float32、宽色域、不裁切、嵌 ICC）",
                            description="产物合同与色彩工作包（docs/COLOR_CONTRACT_PACKAGE_20260922.md）：M 后期母版。")
    master.add_argument("--source", required=True, help="作业目录、products/<uuid> 目录或 master_linear.fit / sky_master.fit")
    master.add_argument("--output", default="outputs", help="根目录，内建唯一新目录 master-<uuid>")
    master.add_argument("--reference-dng", help="参考帧：DNG 读 BlackLevel/WhiteLevel/ColorMatrix/AsShotNeutral；其他 RAW 与作业路径一样走 LibRaw 元数据（参数名沿用旧名）")
    master.add_argument("--recipe", help="已有 raw_color 配方 JSON（与 --reference-dng 二选一）")
    master.add_argument("--wb", choices=("daylight", "as_shot"), default="daylight", help="白平衡模式：日光型中性（矩阵推出）或参考帧机内值")
    master.add_argument("--gamut", choices=("rec2020", "prophoto", "srgb"), default="rec2020", help="母版线性色域")
    master.add_argument("--dcp-profile", help="Z8：用固定 D65 DCP ForwardMatrix 代替 LibRaw 矩阵（仅矩阵，不用 HueSatMap）")
    render = sub.add_parser("render-photo", help="从 M 后期母版用唯一的显示变换另存成片（D：16 位 sRGB TIFF + 同源 JPEG）",
                            description="产物合同与色彩工作包：D 成片；--candidates 另存供用户挑选的缩小候选集。")
    render.add_argument("--master", required=True, help="master_scene_linear.tif 或其目录")
    render.add_argument("--output", default="outputs", help="根目录，内建唯一新目录 photo-<uuid>")
    render.add_argument("--sky-mask", help="TOP-DOWN 天空蒙版：锚点只在天空内量，统计按天空/地景分开")
    render.add_argument("--look", choices=("auto", "none", "dcp_hsm"), default="auto",
                        help="观感：auto = 该机型登记的默认观感（与作业成片一致；见 `starstack devices`，未登记观感的机型为无）；"
                             "none = 无；dcp_hsm = Z8 DCP HueSatMap（用户 v11 决定）")
    render.add_argument("--dcp-profile", help="dcp_hsm 观感与候选集对照所需的固定 Z8 配置文件")
    from .photo_render import DEFAULT_RECIPE as _recipe
    render.add_argument("--contrast", type=_finite_range(0., 100.), default=_recipe["asinh_strength"], help="asinh 强度 k（默认配方 photo-default-1）")
    render.add_argument("--saturation", type=_finite_range(0., 2.), default=_recipe["saturation"], help="饱和度（绕亮度缩放色度）")
    render.add_argument("--target", type=_finite_range(.01, .95), default=_recipe["target"], help="锚点映射到的编码 sRGB 值")
    render.add_argument("--black", type=_finite_range(-1e6, 1e6), default=_recipe["black"], help="显示黑位（母版单位；默认物理零）")
    render.add_argument("--anchor", type=_finite_range(-1e6, 1e6), help="显式锚点亮度（母版单位）；缺省量天空/全图中位数")
    render.add_argument("--wb", choices=("as_shot", "daylight", "master"), default=_recipe["wb_mode"],
                        help="成片白平衡（用户决定 U2 默认机内 as_shot；master = 与母版一致）")
    render.add_argument("--no-shadow-smoothing", action="store_true", help="关闭阴影色度平滑（阶段 1 行为）")
    render.add_argument("--shadow-chroma-end", type=_finite_range(0., 1e6), default=0., help="可选：该亮度（母版单位）以下色度收向中性，0 = 关")
    render.add_argument("--candidates", action="store_true", help="另存反差 × 饱和度、另一种 WB、平滑开/关、DCP 观感的缩小候选集")
    render.add_argument("--no-flags", action="store_true", help="不写逐像素标志图（验收需要标志图）")
    for command in ("status", "cancel", "resume"):
        p = sub.add_parser(command)
        p.add_argument("job")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            result = doctor(args.siril)
        elif args.command == "demo":
            result = {"input_dir": create_demo(args.output), "synthetic": True}
        elif args.command == "inspect-raw":
            from .raw_audit import inspect_raw
            result = inspect_raw(args.source, args.output)
        elif args.command == "protect-trace":
            from .transients import record_candidate
            result = record_candidate(args.audit, args.output, rectangle=args.rectangle,
                                      classification=args.classification, note=args.note)
        elif args.command == "develop-raw":
            from .raw_develop import develop_raw
            result = develop_raw(args.audit, args.output, recipe_path=args.recipe,
                                 exposure_ev=args.exposure)
        elif args.command == "trace-status":
            from .transients import read_candidate
            result = read_candidate(args.candidate)
        elif args.command == 'coordinate-background':
            from .background_job import render_background
            result = render_background(args.job, args.samples, args.recipe, mode=args.mode,
                                       strength=args.strength, red_gain=args.red_gain, blue_gain=args.blue_gain)
        elif args.command == "finish-photo":
            from .photographic_job import finish_photo
            result = finish_photo(args.source, args.output, args.sky_mask, args.protected_mask,
                mode=args.mode, detail_strength=args.detail_strength, star_strength=args.star_strength,
                ground_strength=args.ground_strength, **_quality_option(args))
        elif args.command == "finish-milkyway":
            from .photographic_job import finish_milkyway
            paired_options = ({"transient_background": args.transient_background}
                              if args.transient_background is not None else {})
            if args.starless_background is not None:
                paired_options["starless_background"] = args.starless_background
            if args.milkyway_evidence is not None:
                paired_options["milkyway_evidence"] = args.milkyway_evidence
            result = finish_milkyway(args.source, args.output, args.sky_mask, args.protected_mask,
                args.milkyway_mask, brightness=args.brightness, contrast=args.contrast,
                structure=args.structure, color=args.color, strength=args.strength, **paired_options,
                **_quality_option(args))
        elif args.command == "starless":
            from .starless_runtime import starless as run_starless, NOX_INTRA_OP_THREADS
            # EFF-NOX: the CLI defaults to the workflow's explicit count; the library default stays None (ORT auto).
            result = run_starless(args.source, args.output, args.sky_mask, args.protected_mask, model_path=args.model,
                                  tile=args.tile, stride=args.stride, stitch=args.stitch, batch=args.batch, layout=args.layout,
                                  channel_order=args.channel_order, value_range=args.value_range, providers=args.provider,
                                  intra_op_threads=NOX_INTRA_OP_THREADS if args.threads is None else args.threads,
                                  transient_background=args.transient_background)
        elif args.command == "export-star-layers":
            from .star_layer_export import export_star_layers
            result = export_star_layers(args.separation_result, args.output)
        elif args.command == "reduce-stars":
            from .star_reduction import reduce_stars
            range_options = ({"background_strength": args.background_strength,
                              "milkyway_offset": args.milkyway_offset}
                             if args.background_strength is not None else {})
            result = reduce_stars(args.separation_result, args.output, strength=args.strength,
                bright_start=args.bright_start, bright_end=args.bright_end,
                milkyway_mask=args.milkyway_mask, milkyway_evidence=args.milkyway_evidence,
                milkyway_relief=args.milkyway_relief, bright_radius=args.bright_radius, **range_options,
                **_quality_option(args))
        elif args.command == "suggest-stars":
            from .star_reduction import suggest_star_reduction
            result = suggest_star_reduction(args.separation_result, args.output,
                milkyway_mask=args.milkyway_mask, milkyway_evidence=args.milkyway_evidence)
        elif args.command == "export-master":
            from .color_products import export_master
            result = export_master(args.source, args.output, reference_dng=args.reference_dng, recipe=args.recipe,
                                   wb_mode=args.wb, gamut=args.gamut, dcp_profile=args.dcp_profile)
        elif args.command == "render-photo":
            from .color_products import render_photo
            result = render_photo(args.master, args.output, sky_mask=args.sky_mask, look=args.look, dcp_profile=args.dcp_profile,
                                  contrast=args.contrast, saturation=args.saturation, target=args.target, black=args.black,
                                  anchor=args.anchor, wb_mode=args.wb, shadow_smoothing=not args.no_shadow_smoothing,
                                  shadow_chroma_end=args.shadow_chroma_end, candidates=args.candidates, write_flags=not args.no_flags)
        elif args.command == "help-guide":
            from .user_guide import guide_text
            print(guide_text())
            return 0
        elif args.command == "workflow":
            from pathlib import Path
            from .workflow import submit_workflow, workflow_status
            options = json.loads(Path(args.adapter).read_text(encoding="utf-8")) if args.adapter else None
            result = submit_workflow(args.input, args.output, siril_path=args.siril, sky_mask_path=args.mask,
                stack_engine_path=args.stack_engine, star_model_path=args.star_model,
                background_strength=args.background_strength, backend_options=options,
                idempotency_key=args.key, stack_range_mode=args.stack_range_mode,
                input_files=args.files, trails_enabled=args.retain_trails, stars_enabled=args.reduce_stars,
                color_adjustments={"warmth": args.warmth, "tint": args.tint, "exposure_ev": args.exposure_ev},
                **({"trail_default": args.trail_default} if args.trail_default else {}),
                **({"save_starless": True} if args.save_starless else {}),
                **({"save_star_layer": True} if args.save_star_layer else {}),
                **_quality_option(args))
            if args.wait:
                while result["state"] not in {"succeeded", "failed", "cancelled"}:
                    time.sleep(.5)
                    result = workflow_status(result["workflow_dir"])
        elif args.command in ("workflow-status", "workflow-cancel", "workflow-resume"):
            from . import workflow as pipeline
            method = {"workflow-status": pipeline.workflow_status, "workflow-cancel": pipeline.cancel_workflow,
                      "workflow-resume": pipeline.resume_workflow}[args.command]
            result = method(args.workflow_dir)
        elif args.command == "run":
            result = submit_job(args.input, args.output, args.siril, args.mask, args.key, frame_masks_dir=args.frame_masks,
                                transient_marks=args.transient_marks, support_mode=args.support_mode,
                                stack_range_mode=args.stack_range_mode, force_restart=args.force_restart,
                                **({"stack_engine_path": args.stack_engine} if args.stack_engine else {}))
            if args.wait:
                while result["state"] not in {"succeeded", "failed", "cancelled"}:
                    time.sleep(.5)
                    result = load_job(result["job_dir"])
        elif args.command == "status":
            result = load_job(args.job)
        elif args.command == "devices":
            from . import devices as _dev
            result = {"problems": _dev.registry_problems(), "ok": not _dev.registry_problems()} if args.check else _dev.registry_listing()
        elif args.command == "inspect-device":
            from . import devices as _dev
            result = _dev.inspect_device(args.source)
        elif args.command == "gc":
            from . import gc as _gc
            result = _gc.apply(args.job, work_cache=args.work_cache) if args.apply else _gc.plan(args.job, work_cache=args.work_cache)
        elif args.command == "transients":
            result = _transients(args)
        elif args.command == "cancel":
            cancel_job(args.job)
            result = {"cancel_requested": True}
        else:
            result = resume_job(args.job)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("state") == "failed" or result.get("available") is False else 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        return 1


def _transients(args) -> dict:
    """List or change a finished job's bright-trail candidates (RUN_LOG §28)."""
    from pathlib import Path
    from .transient_stage import read_record, record_path, summary as transient_summary
    from .transient_review import review_candidate
    labels = {cid: (None if cls == "auto" else cls) for cid, cls in (args.label or [])}
    overrides = {cid: (None if d == "auto" else d) for cid, d in (args.override or [])}
    policy = {}
    for item in args.policy or []:
        if "=" not in item:
            raise ValueError("策略写法：CLASS=keep 或 CLASS=omit")
        cls, decision = item.split("=", 1)
        policy[cls.strip()] = decision.strip()
    marks = [{"input": inp, "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1)} for inp, x0, y0, x1, y1 in (args.mark or [])]
    if getattr(args, "protect", False):
        if not marks:
            raise ValueError("--protect 需配合 --mark；已有候选使用 --override ID keep")
        for mark in marks:
            mark["protect"] = True
    notes = {cid: text for cid, text in (args.note or [])}
    if labels or overrides or policy or marks or notes or getattr(args, "layer_style", None):
        job = recompose_transients(args.job, labels=labels or None, overrides=overrides or None, policy=policy or None,
                                   marks=marks or None, notes=notes or None, layer_style=getattr(args, "layer_style", None),
                                   reviewer="human")
    else:
        job = load_job(args.job)
    record = read_record(Path(job["job_dir"]) / "work", job=job)
    if record is None:
        return {"job_dir": job["job_dir"], "transients": None, "message": "这个作业没有亮迹记录"}
    # Rebuild from authoritative candidates, including old immutable revisions.
    current_summary = transient_summary(record)
    selected = record["candidates"]
    needs_review = getattr(args, "needs_review", False)
    if needs_review:
        by_id = {c["id"]: c for c in selected}
        selected = [by_id[cid] for cid in current_summary["recall_review"]["candidate_ids"]]
    scope = {"candidate_filter": "needs_review" if needs_review else "all",
             "summary_scope": "all_candidates", "returned_candidates": len(selected)}
    artifacts = [a for a in job.get("artifacts", []) if a.get("group") == "transients"]
    if args.full:
        view_record = {**record, "summary": current_summary,
                       "candidates": [{**c, "recall_review": review_candidate(c)} for c in selected]}
        return {"job_dir": job["job_dir"], "transients": view_record, "artifacts": artifacts, **scope}
    return {"job_dir": job["job_dir"], "summary": current_summary, "policy": record.get("policy"),
            "related_candidate_groups": record.get("related_candidate_groups"), "application_by_source": record.get("application_by_source"),
            "candidates": [candidate_view(c) for c in selected if (c.get("measurement") or {}).get("status") == "measured"],
            "not_measured": [candidate_view(c) for c in selected if (c.get("measurement") or {}).get("status") != "measured"],
            "review_order": current_summary["recall_review"]["candidate_ids"] if needs_review else None,
            "artifacts": artifacts, **scope,
            "record": str(record_path(Path(job["job_dir"]) / "work", job=job))}


# The compact candidate view lives in transient_review (a leaf module); re-exported here for existing callers.
from .transient_review import candidate_view  # noqa: E402,F401


if __name__ == "__main__":
    raise SystemExit(main())

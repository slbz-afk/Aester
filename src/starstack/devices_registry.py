"""Device support registry (docs/PRE_RELEASE_CLOSURE_20260922.md D, docs/DEVICE_SUPPORT_20260922.md).

Plain Python data so it travels with ``src/starstack/*.py`` into the trial package.  Every entry states what was
actually verified and what would raise it to the next tier; nothing here is a promise that a camera "works".

Tiers
  A   real sequence job(s) accepted through the product (stack + composite + M/D contract where applicable)
  B+  real single frames checked against the decode/metadata contract (black/white, matrix, as-shot WB, geometry,
      compression variant); no sequence acceptance
  B   declared support only: in the decoder's supported list, no camera-specific code path, no real file checked
  C   known problem / not supported
Cameras not listed are "unregistered": handled by the decoders' own support lists, nothing checked by us.
"""
from __future__ import annotations

import copy

REGISTRY_VERSION = "devices-registry-5"          # 3: per-body colour_look; 4: look batch; 5: strengths set by the user

TIERS = {
    "A": "真实序列作业验收通过（堆栈 + 合成 + 适用时的 M/D 合同）",
    "B+": "真实单帧核对过解码/元数据合同（黑白位、矩阵、机内 WB、几何、压缩变体），无序列验收",
    "B": "声明支持：解码器支持列表内、产品无机型特判、未用真实文件核对",
    "C": "已知问题 / 不支持",
}

# What a tier needs to move up; referenced by entries so the CLI can print it.
NEXT = {
    "B": "B+：任意一张真实 RAW（每种压缩变体各一张）跑 `starstack inspect-device`，颜色底子可建立且黑白位/矩阵/WB/几何与厂商规格一致；"
         "A：设备验收拍摄清单（同曝光 ≥ 8 帧星野含地景 + ≥ 3 张同曝光暗场）跑通作业并通过 M/D 验收脚本",
    "B+": "A：设备验收拍摄清单跑通作业并通过 M/D 验收脚本",
    "A": "已是最高级；新固件/新压缩变体需重新核对",
    "C": "先修已知问题",
}

SONY_COMMON = {
    "make": "SONY",
    "container": "tiff",                      # ARW = TIFF container, EXIF Make/Model in IFD0
    "cfa": "RGGB",
    "bits": 14,
    "raw_variants": {
        "uncompressed": {"decoder": "declared", "note": "厂商 14 位未压缩；LibRaw 0.22 列表内"},
        "lossy_compressed": {"decoder": "declared", "note": "厂商 '压缩 RAW'（11+7 位增量编码）；亮星边缘可能有量化伪影，待实测"},
        "lossless_compressed": {"decoder": "declared", "note": "厂商 '无损压缩 RAW'（L/M/S）；目标用户偏好；老版 Siril 曾无法转换 a7 IV 无损 ARW，1.4.4 待用真实文件核对"},
    },
    "variants_note": "各机身提供哪些变体按厂商规格页核对（本表未逐台核对）；`inspect-device` 读 SonyRawFileType (0x7000) 判断实际变体",
    "tier": "B",
    "evidence": ["libraw.org/supported-cameras（LibRaw 0.22 列表）", "rawpy 0.27.1 基于 LibRaw 0.22（requirements-lock.txt）"],
    "verified_frames": [],
    "quirks": {},
    "public_samples": "raw.pixls.us（许可栏需逐样核对；沙箱代理不可达，未取样）",
}


def _sony(model_id: str, name: str, mp: float, shape_hw: list[int], star_eater: str, extra_samples: str = "") -> dict:
    # Deep copy: entries must not share raw_variants / verified_frames / evidence objects (docs/SONY_ADAPTATION_PREREG C1).
    entry = {**copy.deepcopy(SONY_COMMON), "model_ids": [model_id], "name": name, "megapixels": mp,
             "shape_hw": shape_hw, "shape_source": "公开规格（未用文件核对）",
             "quirks": {"star_eater": star_eater}}
    if extra_samples:
        entry["public_samples"] = SONY_COMMON["public_samples"] + "；" + extra_samples
    return entry


# α7R V: real files checked (docs/SONY_ADAPTATION_PREREG_20260925.md stage 1).  The user decided on 2026-09-25 that the
# body stays at tier B until it has a camera-specific colour profile, so the evidence is recorded without a tier change.
A7RV_STAGE1 = "α7R V 阶段 1 实测（2026-09-25）"
A7RV_STAGE3 = "α7R V 阶段 3 实测（2026-09-26）"
A7RV_CHECKED = {
    "name": "无损压缩 L（SonyRawFileType 4，CFA）",
    "raw_file_type": 4, "cfa": True, "raw_ifd_wh": [9728, 6656], "firmware": "ILCE-7RM5 v2.02",
    "libraw_visible_wh": [9566, 6374], "default_crop": {"origin_xy": [32, 20], "size_wh": [9504, 6336]},
    "black_DN": 512, "white_DN": 16383,
    "evidence": A7RV_STAGE1 + "（项目自验；隔离复核已复现；独立验收待）",
    "frames": [
        {"name": "DSC07337.ARW", "sha256": "5dec800858af14b1c5858b02b432954db5c825cd4978634d949d2061f3c78eef"},
        {"name": "DSC07376.ARW", "sha256": "0455840a3e505e35cf95d50d288d69af0d3306923bb409ffee2dfc9221687d92"},
        {"name": "DSC07414.ARW", "sha256": "f58b3d56ba8ec753baf07cb03a6d4899067643ee662f4a26ccf596cf04d31223"},
        {"name": "DSC07606.ARW", "sha256": "6b9b5c917e20b28dc09ffdc2d67386b14ff96ceedd220d779434d52787d4129d"},
        {"name": "DSC07620.ARW", "sha256": "7148861e52d562582a3979e07e98ddbfc9fad41d8ed08e46754c789e80c44802"},
        {"name": "DSC07634.ARW", "sha256": "01ef8626af8ac78360264d5c319c6054da35f402f1dad5afdca6d64b1095bf70"},
    ],
}


def _a7rv() -> dict:
    entry = _sony("ILCE-7RM5", "Sony α7R V", 61.0, [6374, 9566],
                  "公开暗场实测报告不再削孤立亮像素（cloudynights 902676，2023-12）；本产品未测（用户 2026-09-25 决定不作门槛）")
    entry["shape_source"] = ("LibRaw 0.22.1 可见区实测（与 Siril 1.4.4 转换帧逐点相同）；厂商 L 尺寸 9504×6336 为 DefaultCrop，"
                             "RAW IFD 9728×6656 含瓦片补齐")
    entry["checked_subvariants"] = [copy.deepcopy(A7RV_CHECKED)]
    entry["quirks"]["white_level"] = ("LibRaw white_level 16383 = 容器 WhiteLevel = Adobe DNG Converter 18.6 WhiteLevel；"
                                      "linear_max（0x787f）为 15360，但本批 ISO 800/1600、固件 v2.02 的 118 帧中 472 个饱和平台全部在 16383，"
                                      "产品按 16383 计饱和正确；RawTherapee #7555 的 15360 白位假设在这批素材上不成立")
    entry["quirks"]["edges"] = "LibRaw 可见区 9566×6374 四边无异常（6 帧噪声单位检查）；ART 的 9564×6374 裁切在本批素材上不需要"
    entry["next_note"] = ("升级需机型专用颜色配置（例如自制 DCP）并经用户决定（2026-09-25）；"
                          "暗场不作门槛（用户 2026-09-25 决定，仅限 α7R V）")
    entry["public_samples"] = "用户提供（2026-09-25），许可未确认；不得打包、训练或作对外样例"
    entry["evidence"] = [*entry["evidence"], A7RV_STAGE1 + "：无损 L 单帧解码/元数据核对（Adobe 独立解码与 Siril 均零差），项目自验",
                         A7RV_STAGE3 + "：正式工作流（MCP 共用路径）三个真实作业均成功，M/D 色彩合同（S2、M1、M4–M7、M9e）"
                         "均通过（c1 自身的 M9/M8 因 Z8 旧图几何不同而判 FAIL，只报告），竖拍方向正确；30 秒不跟踪，"
                         "Siril 主轴 FWHM 中位数 26 mm 约 16 px（3a/3b）、16 mm 约 9.6 px（3c），配准回退 Siril 两遍，"
                         "参与帧 5/5、25/29、13/78（素材质量所致，用户 2026-09-26 决定不另立项）",
                         "α7R V 隔离复核（2026-09-26）：复核者用自己的代码复现单帧解码零差、黑白位/矩阵/白平衡，"
                         "以及三个作业的 S2、Q4、Q6、Q7（不是独立验收；独立验收待）"]
    entry["quirks"]["registration"] = ("30 秒不跟踪，星点沿赤经拖长（Siril 主轴 FWHM 中位数：26 mm 约 16 px、16 mm 约 9.6 px），"
                                       "每帧约 130–240 颗星，产品自有畸变标定不可用、回退 Siril 两遍配准，远离参考的帧易被跳过；"
                                       "属素材条件（" + A7RV_STAGE3 + "）")
    entry["quirks"]["dng_route"] = ("只核对了原生 ARW 路线。Adobe DNG Converter 18.6 转出的 DNG 带 CameraCalibration "
                                    "diag(0.9337, 1, 1.0124)（签名 com.adobe），产品 DNG 颜色路线会拒绝（CameraCalibration 非单位）；"
                                    "产品兼容导入转 DNG 时还会因可见区 6376×9570 ≠ 6374×9566 在几何检查处停止")
    entry["quirks"]["lateral_ca"] = "Adobe DNG 内嵌横向色差校正（R/B 相对 G 在角落约 1–2 px）；产品未应用，星点边缘可见色边"
    # Default finishing look (docs/SONY_A7RV_LOOK_PREREG_20260926.md): no α7R V profile exists, so the user chose on
    # 2026-09-26 to borrow the α7 IV table at half strength.  Borrowed, so it is not a camera-specific profile: tier stays B.
    entry["colour_look"] = {"id": "sony-ilce7m4-rt-cc0-d50", "strength": 0.5,
                            "decided": "用户 2026-09-26：不追求像索尼；饱和度介于默认与借用表之间，偏绿偏紫倾向减半，夜景余色用白平衡微调",
                            "evidence": "docs/SONY_LOOK_TRIAL_20260926.md", "borrowed": True,
                            "status": "用户 2026-09-26 看图接受，维持 B"}
    return entry


# Per-body default looks, batch of 2026-09-26 (docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md B): bodies with an openly
# licensed own DCP (tiers 1–2 of docs/CAMERA_ADAPTATION_CANDIDATES_20260926.md) use its daylight HueSatMap; popular bodies
# without one borrow a table.  Strength 0.5 and the donors are provisional until the user has reviewed the comparisons.
LOOK_BATCH = "docs/CAMERA_LOOKS_AND_PARALLEL_PREREG_20260926.md"
# The user reviewed outputs/camera-looks-batch-20260926/review/review.html on 2026-09-27 and set the strengths below;
# bodies the user did not mention keep the provisional 0.5.
SET_BY_USER = "用户 2026-09-27 看对照页定为 {strength:g}"
KEPT_BY_USER = "用户 2026-09-27 看对照页未要求修改，保持 0.5"


def _status(strength):
    return KEPT_BY_USER if strength is None else SET_BY_USER.format(strength=strength)


def _own_look(look_id: str, strength: float | None = None) -> dict:
    return {"id": look_id, "strength": 0.5 if strength is None else strength, "borrowed": False, "evidence": LOOK_BATCH,
            "status": _status(strength),
            "decided": "用户 2026-09-26：机型表第一、二档按本机开放许可 DCP 适配；只叠色相/饱和度表"}


def _borrowed_look(look_id: str, donor: str, reason: str, strength: float | None = None) -> dict:
    return {"id": look_id, "strength": 0.5 if strength is None else strength, "borrowed": True, "donor": donor,
            "donor_reason": reason, "evidence": LOOK_BATCH, "status": _status(strength),
            "decided": "用户 2026-09-26：热门且无开放 DCP 的机型借表；2026-09-27 供表机维持同传感器机型"}


def _sample_note(sample: int) -> str:
    return (f"观感样片核对 raw.pixls.us CC0 rpu-{sample}：机型识别、LibRaw 颜色底子与默认观感（{LOOK_BATCH}）；"
            "不是 B+ 的元数据合同核对")


NIKON_NEF = {"nef": {"decoder": "declared", "note": "NEF 无损/有损压缩与未压缩；LibRaw 0.22 列表内；HE/HE★ 不适用"}}
CANON_CR3 = {"cr3": {"decoder": "declared", "note": "CR3 RAW；C-RAW 未核对"}}
CANON_CR2 = {"cr2": {"decoder": "declared", "note": "CR2 RAW；sRAW/mRAW 不适用"}}


def _body(make: str, model_id: str, name: str, container: str, mp: float, shape_hw: list[int], sample: int,
          variants: dict, look: dict) -> dict:
    return {"make": make, "model_ids": [model_id], "name": name, "container": container, "cfa": "RGGB", "bits": 14,
            "megapixels": mp, "shape_hw": shape_hw, "shape_source": f"LibRaw 0.22.1 可见区（raw.pixls.us CC0 样片 rpu-{sample}）",
            "raw_variants": copy.deepcopy(variants), "tier": "B",
            "evidence": ["LibRaw 0.22.1 支持列表", _sample_note(sample)],
            "verified_frames": [], "quirks": {}, "public_samples": f"raw.pixls.us CC0 rpu-{sample}",
            "colour_look": look}


def _sony_look(entry: dict, sample: int, shape_hw: list[int] | None, look: dict) -> dict:
    entry["colour_look"] = look
    entry["evidence"] = [*entry["evidence"], _sample_note(sample)]
    entry["public_samples"] = entry["public_samples"] + f"；raw.pixls.us CC0 rpu-{sample}（观感样片）"
    if shape_hw is not None:
        entry["shape_hw"] = shape_hw
        entry["shape_source"] = f"LibRaw 0.22.1 可见区（raw.pixls.us CC0 样片 rpu-{sample}）"
    return entry


REGISTRY: list[dict] = [
    {
        "make": "NIKON CORPORATION", "model_ids": ["NIKON Z 8"], "name": "Nikon Z 8", "container": "tiff",
        "cfa": "RGGB", "bits": 14, "megapixels": 45.7, "shape_hw": [5504, 8256],
        "shape_source": "公开规格；Siril 转换帧实测镶嵌区 5520×8280（含边缘）",
        "raw_variants": {
            "nef_lossless": {"decoder": "verified", "note": "NEF 无损压缩：早期原型作业"},
            "nef_he_star": {"decoder": "workaround", "note": "HE★ 高效压缩：当前需先转 CFA DNG 工作副本（README 使用边界）"},
            "dng": {"decoder": "verified", "note": "Adobe 转换的 CFA DNG：48 帧基线与 8 帧 M/D 验证作业的输入"},
        },
        "tier": "A",
        "evidence": ["docs/COLOR_CONTRACT_PACKAGE_20260922.md（48 帧基线作业 M/D 验收）",
                     "8 帧验证作业 S2a–c（2026-09-22）",
                     "docs/Z8_METEOR_TEST_20260916.md"],
        "verified_frames": ["frame00001–00008.dng（8 帧作业）"],
        "quirks": {"black_level": "1008 ×4（DNG BlackLevel）", "white_level": 15892,
                   "baseline_exposure": "+0.2 EV 未应用（记录）"},
        "public_samples": "不需要：自有素材",
    },
    {
        "make": "ZWO", "model_ids": ["ZWO Seestar S30 Pro"], "name": "Seestar S30 Pro（FITS CFA 单帧）", "container": "fits",
        "cfa": "GRBG", "bits": 16, "megapixels": 8.3, "shape_hw": [3840, 2160],
        "shape_source": "fits_input 已验证组合（PROGRAM 8.46，WIDECAM 0）",
        "raw_variants": {"fits_cfa_uint16": {"decoder": "verified", "note": "缺 ROWORDER 时按已验证组合暂定 TOP-DOWN"}},
        "tier": "A",
        "evidence": ["docs/STACK_INPUT_VALIDATION_20260920.md（16 张 CFA 直接输入，15 参与）"],
        "verified_frames": ["NGC281 序列"],
        "quirks": {"roworder": "头缺 ROWORDER；绝对相位/天球方向/颜色未认证"},
        "public_samples": "自有 + 作者授权素材",
    },
    {
        "make": "Canon", "model_ids": ["Canon EOS 5D Mark III"], "name": "Canon EOS 5D Mark III", "container": "tiff",
        "cfa": "RGGB", "bits": 14, "megapixels": 22.3, "shape_hw": [3840, 5760],
        "shape_source": "公开规格；README：解码 5796×3870，机内 JPG 5760×3840",
        "raw_variants": {"cr2": {"decoder": "verified", "note": "早期原型作业（data/external/photographingspace 6 张 CR2）"}},
        "tier": "B+",
        "evidence": ["README 已完成的实拍任务 outputs/job-579616d928f2066980480eca", "data/external/SOURCES.md"],
        "verified_frames": ["6 张 CR2（练习数据，不得打包或训练）"],
        "quirks": {},
        "public_samples": "练习数据仅限本地使用（SOURCES.md）",
        "next_note": "M/D 合同未在 CR2 上验证（LibRaw 元数据路线）",
    },
    {
        "make": "Panasonic", "model_ids": ["DC-S5"], "name": "Panasonic Lumix S5（改机）", "container": "rw2",
        "cfa": "RGGB", "bits": 14, "megapixels": 24.2, "shape_hw": [4000, 6000],
        "shape_source": "公开规格（未用文件核对）",
        "raw_variants": {"rw2": {"decoder": "verified", "note": "一次真解码（跟踪摄影练习）"}},
        "tier": "B+",
        "evidence": ["docs/MILKYWAY_X1_VALIDATION_20260921.md（S5 跟踪练习）"],
        "verified_frames": ["21 mm / 240 s 天空 + 独立地景 RW2"],
        "quirks": {"container": "RW2 非标准 TIFF 魔数，`identify_camera` 无法从容器识别机型（报未识别），需 rawpy 元数据"},
        "public_samples": "素材非商用许可，产品用途未放行",
    },
    _sony_look(_sony("ILCE-7M3", "Sony α7 III", 24.2, [4000, 6000],
                     "公开报告：长曝光存在空间滤波（star eater）；本产品未测"),
               2414, None, _own_look("sony-ilce7m3-art-pd-d65", 0.75)),
    _sony_look(_sony("ILCE-7M4", "Sony α7 IV", 33.0, [4672, 7008],
                     "未知：社区讨论不一致，用验收清单的暗场统计实测"),
               6932, None, _own_look("sony-ilce7m4-rt-cc0-d50", 0.75)),
    _sony_look(_sony("ILCE-7CM2", "Sony α7C II", 33.0, [4672, 7008],
                     "未知：待实测", "raw.pixls.us 2023-11 收录 5 张（discuss.pixls.us/t/39621）"),
               6868, None, _borrowed_look("sony-ilce7m4-rt-cc0-d50", "SONY ILCE-7M4", "与 α7 IV 同为 33 MP 传感器", 0.75)),
    _sony("ILCE-7CR", "Sony α7CR", 61.0, [6336, 9504],
          "未知：待实测"),
    _sony("ILCE-7RM3", "Sony α7R III", 42.4, [5304, 7952],
          "公开报告：长曝光存在空间滤波（star eater）；本产品未测"),
    _a7rv(),
    _sony_look(_sony("ILCE-7SM3", "Sony α7S III", 12.1, [2848, 4256], "未知：待实测"),
               5418, [2848, 4256], _own_look("sony-ilce7sm3-rt-pd-d65", 0.75)),
    _sony_look(_sony("ILCE-6400", "Sony α6400（APS-C）", 24.2, [4024, 6024], "未知：待实测"),
               3151, [4024, 6024], _own_look("sony-ilce6400-rt-pd-d50", 0.75)),
    _body("NIKON CORPORATION", "NIKON Z 6", "Nikon Z 6", "tiff", 24.5, [4040, 6064], 3582, NIKON_NEF,
          _own_look("nikon-z6-rt-pd-d65")),
    _body("NIKON CORPORATION", "NIKON Z 6_2", "Nikon Z 6II", "tiff", 24.5, [4040, 6064], 4160, NIKON_NEF,
          _own_look("nikon-z6ii-rt-pd-d65")),
    _body("NIKON CORPORATION", "NIKON Z 7", "Nikon Z 7", "tiff", 45.7, [5520, 8288], 2789, NIKON_NEF,
          _own_look("nikon-z7-rt-pd-d65")),
    _body("NIKON CORPORATION", "NIKON Z 7_2", "Nikon Z 7II", "tiff", 45.7, [5520, 8288], 4213, NIKON_NEF,
          _borrowed_look("nikon-z7-rt-pd-d65", "NIKON Z 7", "与 Z 7 同为 45.7 MP 传感器", 0.75)),
    _body("NIKON CORPORATION", "NIKON D850", "Nikon D850", "tiff", 45.7, [5520, 8288], 1840, NIKON_NEF,
          _own_look("nikon-d850-rt-pd-d65", 0.65)),
    _body("NIKON CORPORATION", "NIKON D750", "Nikon D750", "tiff", 24.3, [4032, 6032], 898, NIKON_NEF,
          _own_look("nikon-d750-art-pd-d65")),
    _body("Canon", "Canon EOS 6D", "Canon EOS 6D", "tiff", 20.2, [3670, 5496], 1427, CANON_CR2,
          _own_look("canon-eos6d-art-pd-d65")),
    _body("Canon", "Canon EOS 5D Mark IV", "Canon EOS 5D Mark IV", "tiff", 30.4, [4502, 6744], 983, CANON_CR2,
          _own_look("canon-eos5dmk4-rt-pd-d65", 0.75)),
    _body("Canon", "Canon EOS R", "Canon EOS R", "cr3", 30.3, [4498, 6742], 4515, CANON_CR3,
          _own_look("canon-eosr-rt-pd-d65", 0.75)),
    _body("Canon", "Canon EOS RP", "Canon EOS RP", "cr3", 26.2, [4180, 6264], 4674, CANON_CR3,
          _own_look("canon-eosrp-rt-cc0-d50")),
    _body("Canon", "Canon EOS R5", "Canon EOS R5", "cr3", 45.0, [5463, 8191], 4694, CANON_CR3,
          _own_look("canon-eosr5-rt-pd-d65", 0.4)),
    _body("Canon", "Canon EOS R6", "Canon EOS R6", "cr3", 20.1, [3670, 5496], 4662, CANON_CR3,
          _own_look("canon-eosr6-rt-pd-d65", 0.75)),
    _body("Canon", "Canon EOS R8", "Canon EOS R8", "cr3", 24.2, [4024, 6022], 6585, CANON_CR3,
          _own_look("canon-eosr8-rt-cc0-d50")),
    _body("Canon", "Canon EOS R6m2", "Canon EOS R6 Mark II", "cr3", 24.2, [4024, 6022], 6402, CANON_CR3,
          _borrowed_look("canon-eosr8-rt-cc0-d50", "Canon EOS R8", "与 R8 同为 24.2 MP 传感器")),
]
# α7R IIIA shares the α7R III pipeline
next(e for e in REGISTRY if e["model_ids"][0] == "ILCE-7RM3")["model_ids"].append("ILCE-7RM3A")


def entries() -> list[dict]:
    return [dict(e) for e in REGISTRY]

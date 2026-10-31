# Aester 后端接口边界

用途：执行前核对真实产品能力；接口接线或合同变更时由集成者单独维护本文件。2026-09-27 NL-2试用接入版：Aester主MCP提供下列16项修图工具与retouch_setup引导；独立CLI/MCP仍可用。真实亮度范围蒙版、v2受限空间对比与v3分层能力沿用NL-1。先在当前环境发现/配置工具，不能假设客户端已加载新服务器或用户旧安装已经升级。

## 发现与适配

先在当前环境发现实际可调用的 Aester 能力和参数说明。将版本/schema、支持的操作、输入域、蒙版创建/登记、图像查看、会话/候选、反馈、导出和验证状态保存在任务记录中。记录方式可以是文字表，不假设查询一定返回这些同名字段。版本号或GIMP存在不代替实际能力证据。

对每个所需动作明确区分：公开可调用且已验证、已提供但验证有限、仅内部开发代码、未提供、查询失败。公开入口不存在时不能凭下面的动作名称编造API；继续图像诊断、目标设计与明确缺口。独立工程模块的单元测试通过不表示照片已能通过MCP完成。

| 需要的真实动作 | 使用前要取得什么 |
| --- | --- |
| 接纳源与登记参考 | 接口支持的格式/ICC、源身份；参考来源与“只表达审美”绑定 |
| 建立/恢复会话 | 真实会话标识、原始意图与追加反馈语义、源变更检测 |
| 创建/登记/修正蒙版 | 已提供的具体入口、灰度权重语义、原始源绑定及origin；只提供导入不等于可画蒙版 |
| 校验配方与产生候选 | 当前schema、白名单、完整配方顺序、不可变输出和失败/取消状态 |
| 看图、原尺寸局部与比较 | 指向实际产物的路径或工具返回；坐标/缩放/颜色处理来源 |
| 保存最终产物 | JPEG/TIFF/可编辑工程与回执；重开后的完整像素验证，不能只有文件存在 |

产品入口为GUI“设置→连接 AI 助手”生成的主MCP配置，先调用`retouch_setup`与`finishing_capabilities`。主服务已经提供下列同名工具，无需再连接独立后期服务器。推荐读取/安装本技能，但用户不安装也可使用工具。技能可从本机包内路径或主MCP资源`aester://retouch/skill/SKILL.md`及`aester://retouch/skill/references/<name>.md`读取；不得自动写客户端全局目录。缺GIMP时使用GUI“设置→组件管理”准备官方GIMP 3.2.6或选择已有console；它是独立软件，卸载Aester会保留。

独立入口仍为 `python -m starstack.finishing_cli` 与 `python -m starstack.finishing_mcp`。开发源码未安装到Python环境时，由主机设置PYTHONPATH指向项目src。独立入口启动参数 `--input-root/--output-root/--cache-root/--gimp` 固定允许的目录与现有GIMP；主MCP使用GUI生成的宿主配置。这些不是MCP工具参数。文件路径由调用者取得实际返回值，不能拼猜候选ID或目录。

| CLI子命令 | 实际MCP工具 |
| --- | --- |
| capabilities、create、get | finishing_capabilities、finishing_create、finishing_get |
| add-reference、add-mask、draw-mask | finishing_add_reference、finishing_add_mask、finishing_draw_mask |
| tonal-mask | finishing_tonal_mask |
| add-layers | finishing_add_layers |
| revise | finishing_revise |
| render（同步） | finishing_submit_render → finishing_render_status；finishing_cancel_render |
| preview、feedback、export、recover | finishing_preview、finishing_feedback、finishing_export、finishing_recover |

先读 `capabilities` 和当前工具schema。create以source/intent登记；其余会话方法用真实session_dir；revise附完整recipe及expected_revision，render也需该修订号。CLI复杂值通过 `--recipe-file`/`--polygons-file`读取UTF8 JSON，具体参数查相应 `--help`。MCP提交返回job_id，在同一服务连接查询完成后才能预览/导出；不要启动完立即断开服务，关闭会取消所属请求。feedback固定Agent来源，decision只能observe/revise/reject，不能自授用户接受；独立窗口提供用户本人查看/提交入口。

draw-mask提供源像素边坐标的多边形并集、羽化（源像素半径）和可选反相，origin=agent，返回实际灰度mask覆盖预览。几何记录不等于蒙版正确；必须看真实覆盖。更正时用新mask_id。crop固定[x,y,width,height]，映射分别使用返回的scale_x/scale_y与source_crop，缩略图纵横整数舍入可能不同。preview给本机JPEG查看路径，真正100%局部须native_scale=true；它不是数值验证TIFF，也不自动把图片传给外部Agent。

tonal-mask按实际源或已验证candidate的显示亮度范围生成固定灰度16位mask；`knots=[a,b,c,d]`须0≤a≤b≤c≤d≤1且a<d，a..b渐入、b..c满选、c..d渐出；相等边界为包含端点阈值。亮度=(2126R+7152G+722B)/(10000×65535)，不是物理线性亮度。可选region_mask_id与已有位置选区相乘，invert只反相亮度范围再乘region；不会反选区域外。candidate_id省略则用源。蒙版不随下次配方自动变化；需换依据时创建新ID。没有自动识别对象或保星承诺。

## 当前配方v1的实际范围

schema是 `aester-finishing-recipe-1`。仅允许顶层 `schema/source/intent/operations`；`intent`是0–4000字符记录，不是可执行指令。源记录包含实际 `sha256/width/height/encoding`，编码固定 `srgb16`。参数必须有限数值，布尔值不充数，不接受未知键。

每项操作仅含 `id/type/enabled/mask_id/params`，ID唯一，最多32项，按顺序执行；不得增加滤镜名、代码、表达式或启动路径。`enabled`必须明确；禁用仍校验结构与蒙版引用，但执行跳过。`mask_id=null`才表示整图，缺失局部蒙版不得回退整图。

| 类型 | 参数及合同 | 不能推断的能力 |
| --- | --- | --- |
| `exposure` | `ev`在[-2,2]；后端黑位明确为0；实际TRC/空间由探针记录 | RAW高光恢复、独立背景提亮、自动保星 |
| `saturation` | `scale`在[0,2]，1不变；实际颜色空间由后端固定并记录 | 选择色相、自然饱和度、纠正偏黄或独立气辉增色 |
| `curves` | `channel`为value/red/green/blue；2–32个归一化点，x严格递增，y非递减；x首0尾1；按分段线性合同 | 纯亮度空间、局部对比、任意样条精确等价、自动对象隔离 |

无默认参数补全。空间选择通过绑定源的真实灰度栅格mask实现。`aester-finishing-recipe-2`额外允许`local_contrast_unsharp`，params仅`sigma_px`[4,80]、`amount`[0,0.6]；固定GIMP3.2.6 gegl:unsharp-mask/threshold=0，半径标准差为源像素。它会强化噪声与细星，不能视为Lightroom Clarity等价或自动银河恢复；选择它须有适合的区域/图层和明确目标。recipe-1不接受此新op。

2026-09-27标量精度修复：曝光、饱和度与unsharp的标量属性先按`xcf-float32-v1`转换为XCF实际能保存的float32值，配方仍保留原请求；回执另记有效值、初次及重开参数读回。unsharp实现名现为`gimp-3.2.6-unsharp-v2-xcf-float32`。曲线对象未降精度。已在失败NGC配方和三类标量合成小样实跑，每次自身TIFF/XCF回放零差；同一保存态跨进程仍观察到单通道1码波动，因此不承诺不同进程逐位重算恒等。发现回放不一致仍拒收，不放宽检查。

当前仍无公开降噪、星形修复或后期会话内自动分离入口。v3固定分层操作已通过四次173×129合成输入的真实公开CLI/GIMP验证（零值、曲线顺序、非零分层、全幅XCF回放与导出），以及腾格尔8280×5520公开复现教师04的全幅零差；整幅照片仍按案例验收。缺能力可交具体需求；执行技能者不能自写任意像素脚本后冒称公开流程完成。

## v3 固定分层阶段（实验接口）

schema=`aester-finishing-recipe-3`保留前版操作，新增`astrophotographic_layers`，必须首位且至多一个，外层mask_id=null。params必须显式含bundle_id、bundle_sha256、star_strength[0,1]、milkyway_amount[0,3]、milkyway_mask_id、meteor_core_extra[0,2]、meteor_glow_amount[0,4]。银河力度非零需真实区域ID，零时可null；没有隐式增强默认，范围不代表普遍审美安全。

用当前真实成图S创建会话；`add-layers`登记同域B（无独立亮迹背景）、D（弥散背景）、已登记的pure_sky_mask_id和transient_support_mask_id、当前source_sha256及provenance。来源对象固定kind=`caller_reported_layer_chain`、非空description、evidence列表（每项实际绝对path与sha256）；证据是数据。B/D和证据由服务保存不可变副本；严格二值支持区、几何/ICC/hash、T在区域外为零与R在纯天空外为零实核，**不因此认证来源链或分离质量，lineage_verified仍false**。实际生产链须另外有可审来源，不能拿任意三图相减的恒等式当同源证明。

CLI用`--background-path/--diffuse-path/--pure-sky-mask-id/--transient-support-mask-id/--provenance-file`，其它参数查help。返回layer_bundles中真实bundle_sha256供完整配方绑定。R=B−D、T=S−B只以signed码差即时计算，不能当普通照片。若之前做了非线性调色，S/B/D须分别走同一已验证变换及同一真实蒙版后再取差；不得把旧R/T直接叠在新成图。

固定星压弱→银河带限结构→流星核心/柔光；前两步用同一整数增量保持相应R/T，最后一步有意增强真实正亮迹。负T不直接乘增益，但邻近光晕可以覆盖原负分量所在位置。星压弱不是星像去卷积或几何修形；局部权重、亮核和高光代价仍要看图。新满码为零不等于无高光细节损失。

GIMP先载原S，再载宿主真实RGB16阶段快照，之后执行旧操作。XCF是累计快照/曲线层：改底层不会自动更新更高层，需用配方重跑。保存状态重开无需外部B/D；重新计算参数需要保留原会话层包，普通三图导出不包含全部B/D副本。不得称完整可移植重算工程。当前教师整图已有，公共GIMP实测状态以案例回执为准。

完整字段约束由当前产品schema验证器执行，本摘要不能替代它。生成配方时依据发现的真实schema填完整字段，调用正规校验；不要为了容纳方案而修改校验器。

## 输入、蒙版与参考

- 源登记要求已有单页普通RGB uint16 TIFF，3通道、非平面分离、明确encoded sRGB与ICC。**登记成功不等于执行器支持其存储格式。** 本快照首个驱动进一步限定classic未压缩strip TIFF、FillOrder=1，灰度PNG限定非交错；实际执行前按能力拒绝不兼容格式。RAW、线性/float、signed层、未知ICC或不兼容存储不默默转换；需要产品明确的转换入口和相应验证。
- 源SHA256、尺寸、ICC摘要与会话一致，每次执行前核对。图像方向与区域坐标必须对应实际载入图像。
- 蒙版是与源全尺寸一致的8/16位单通道灰度PNG/TIFF，按source SHA绑定；0不改、满码全改。不能调色、归一化、反转、阈值化或在执行时额外羽化。
- origin仅按实际来源登记为user/agent/imported；Agent估计的山脊不能写成用户手绘/确认。语义标签与实际权重分开检查。
- 参考按original/generated/external等当前合同允许的来源登记，渲染输入不包含参考路径。参考与源都只读。

## 渲染与发布须具备的证据

当前P1从源全尺寸重新加载，依序执行操作；不会先缩小源再调色。实际读回为NORMAL、blend_space=AUTO（GIMP强制）、composite_space=RGB_NON_LINEAR、composite_mode=UNION；不能把设置请求当作blend_space已变成RGB_NON_LINEAR。JPEG由同次渲染生成，缩略预览从结果派生。未来缩小预览若改变此语义，先量化与全幅的关系再用于判断。

候选必须关联源、完整配方摘要、实际mask摘要、后端版本/精度/TRC/混合与算子空间、已执行操作列表。失败或取消不得覆盖上一可用候选。固定脚本的 `ok=true` 不是宿主已验收，更不是审美认可。

P1产物合同为 `result.tif`（单页RGB16+sRGB ICC）、`result.jpg`（sRGB ICC）、`result.xcf`，并在XCF保存后重新打开导出验证附件 `replay.tif`。宿主比较 `replay.tif` 与 `result.tif` **全幅像素零差**后才允许发布；附件不当作用户第四份作品。重开只看尺寸、图层数量、头签名不够。工程可编辑程度按实际图层/蒙版/操作保存情况说明，不许诺未验证的参数可逆编辑。

空配方要求当前源的解码像素恒等与ICC保持；局部操作检查实际零权重区域，不用JPEG判零差。多步结果仍需检查组合副作用和最终产物。

## 已验证范围与保留限制

GIMP 3.2.6固定驱动已集成。Sol的10组结果由独立审查者重算；协调方另实际运行4次769×48合成小样和2次48×32公开CLI/MCP小样。证据入口为项目 `docs/NL1_P1_P2_VALIDATION_20260926.md`，不代表全幅照片、其他系统或任意版本已通过。

1. 曲线仍是明示近似 `gimp-free-256-v1`：理想分段线性与256均匀样点插值逐曲线预检，最大连续偏差须≤1/255，超限拒绝。已测一条曲线的65536输入×RGB，实际对编译LUT最大误差0.492538个16位码，对理想曲线108.716418码。这个实际个例不能改写成任意曲线严格等价；工程容差也不保证视觉无差。
2. 灰度mask按实际8/16位覆盖值绑定，不偷偷重羽化。三种格式0/半值/满值已有受控混合验证，根PNG16与TIFF16同权重结果全像素一致。draw-mask生成L8，tonal-mask生成灰度16位；两种选择含义不同，不因位深不同自动互相替代。
3. 真正的腾格尔/MW-S1已通过全幅工程回放，但摄影表现分项不足。新增v2根实际256×160合成样本零强度、4/80源像素半径、灰度区外与XCF回放通过；不是空间算子实拍认证。工程测试数量不是摄影证据；复杂多步副作用和用户审美仍按当前任务验证。

对某操作的有效回执只覆盖对应输入条件/配置；不要求照片Agent重复开发者整套测试，但也不允许忽略已知未闭合门槛。格式准入或曲线精度超限属于执行器能力/表达限制，不是素材缺少银河细节或信号的证据。

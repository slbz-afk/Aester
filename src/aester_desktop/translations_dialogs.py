# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Wang Jian (王俭). See LICENSE.txt in this folder.
"""English copy for region editing, trail review and optional advanced tools."""
EN = {
    '使用这些分区': 'Use these regions', '取消': 'Cancel',
    '天空、地景与银河范围': 'Sky, foreground and Milky Way regions',
    '完成地平线': 'Finish horizon', '寻找附近边缘': 'Find nearby edge',
    '我确认：所圈区域存在可见银河（不确定时不要勾选）': 'I confirm a visible Milky Way in this region (leave unchecked if unsure)',
    '拖出银河椭圆': 'Draw Milky Way ellipse', '接受浅蓝色边缘': 'Accept light-blue edge',
    '撤销': 'Undo', '擦回天空': 'Restore sky', '整幅都是天空': 'Entire frame is sky',
    '方向': 'Angle',
    '沿地平线点选': 'Mark horizon', '画笔大小': 'Brush size', '补涂地景': 'Paint foreground',
    '请先沿地平线点选，并点击“完成地平线”。': 'Mark points along the horizon, then choose Finish horizon.',
    '请完成地景分区，或明确选择整幅都是天空。': 'Finish the foreground region, or confirm that the entire frame is sky.',
    '边缘辅助只在所画地平线附近寻找明暗变化，不能自动消除树叶运动或反光。': 'Edge assistance follows nearby brightness changes. It cannot correct moving leaves or reflections.',
    '适合窗口': 'Fit to window', '重做': 'Redo', '银河羽化': 'Milky Way feathering',
    '浅蓝虚线是待接受的建议。确认它沿真实地平线后再接受；也可以继续使用原线。': 'The dashed light-blue line is a suggestion. Accept it only if it follows the real horizon; you can keep your original line.',
    '\n只有明确选择保留才会加入新成片；自动建议不代表已采用。': '\nOnly an explicit Keep choice adds a trail to the new image; suggestions are not applied automatically.',
    '\n已记录身份：': '\nRecorded identity: ',
    '\n未能测量定位，不能把省略视为已排除流星': '\nPosition could not be measured. Omission does not rule out a meteor.',
    ' · 尚未保存': ' · Unsaved', ' · 草稿已保存': ' · Draft saved',
    '{0} 条有本次选择': '{0} candidates have review choices',
    '{0} · {1}\n已记录身份：{2}；复核处置：{3}。 ': '{0} · {1}\nRecorded identity: {2}; review decision: {3}. ',
    '亮迹复核 · Aester': 'Review trails · Aester', '伪影': 'Artifact',
    '保存复核草稿': 'Save review draft', '保存草稿': 'Save draft', '保留': 'Keep',
    '候选 / 曝光': 'Candidate / exposure',
    '先选择候选和要更改的身份或处置。': 'Select candidates, then choose an identity or a keep/omit decision.',
    '关闭': 'Close', '关闭前保存这些选择？': 'Save these choices before closing?',
    '前一曝光': 'Previous exposure', '卫星': 'Satellite', '后一曝光': 'Next exposure',
    '处置状态': 'Decision', '处置：保持原样': 'Decision: unchanged',
    '应用全部选择并出图': 'Apply choices and render', '建议身份': 'Suggested identity',
    '张 · 候选': ' · Candidate ', '待定': 'Pending', '恢复默认省略': 'Reset to default omission',
    '撤销上一步': 'Undo last change', '撤销选中项的本次选择': 'Clear choices for selected items',
    '未知': 'Unknown', '未选择': 'Not selected',
    '本任务没有可用的整图预览': 'No full-frame preview is available for this job.',
    '本次扫描没有候选；这不等于原片不存在未被检测到的流星。': 'No candidates were found in this scan. Undetected meteors may still be present in the originals.',
    '本次曝光': 'Source exposure', '本次选择': 'Your choice',
    '此条没有可靠的测量位置': 'No reliable measured position is available.',
    '流星': 'Meteor', '清除已有人工身份': 'Clear manual identity', '清除身份': 'Clear identity',
    '省略': 'Omit', '第': 'Frame ', '设置选中项': 'Apply to selected',
    '读取实际曝光…': 'Loading source exposures…', '身份：保持原样': 'Identity: unchanged',
    '选中一条或多条亮迹，分别决定身份与是否保留。只有明确选择“保留”才会加回，判断为流星也不会自动加回。\n未知不等于卫星；自动省略不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。': 'Select trails and choose identity and keep/omit decisions separately. Only Keep adds a trail back; a Meteor label alone does not.\nUnknown does not mean satellite, and automatic omission does not rule out a meteor. Save a draft, then apply all choices. Preservation covers only the measured extent.',
    '选择候选查看来源与邻帧。': 'Select a candidate to see its source and neighboring exposures.',
    '邻帧图使用实际曝光，不将相邻候选误当成相邻曝光。': 'Neighbor views show consecutive exposures, not nearby candidate entries.',
    '飞机': 'Aircraft', '默认省略': 'Omit by default', '；复核处置：': '; review decision: ',
    ' · 可选': ' · Optional', ' · 处置：': ' · Decision: ', '不更改': 'Unchanged',
    '亮迹保护蒙版': 'Trail protection mask', '亮迹复核': 'Trail review', '人工身份': 'Manual identity',
    '仅偏色': 'Color cast only', '仅渐变': 'Gradient only',
    '任务路径已改变，请重新读取候选再复核。': 'The job path changed. Reload its candidates before reviewing.',
    '保存此条复核并重做亮迹层': 'Save this review and rebuild the trail layer',
    '分层结果记录': 'Separation result record', '力度': 'Strength',
    '另存背景协调结果': 'Save background adjustment', '另存银河增强结果': 'Save Milky Way enhancement',
    '图像或记录 (*.tif *.tiff *.png *.json);;所有文件 (*)': 'Images or records (*.tif *.tiff *.png *.json);;All files (*)',
    '处理完成': 'Completed', '处理方式': 'Method', '天空权重蒙版': 'Sky weight mask',
    '完整任务的不可变堆栈快照缺失。': 'The workflow is missing its immutable stack snapshot.',
    '完整出图任务尚未成功结束，不能重做其中的亮迹层。': 'The workflow must finish successfully before its trail layer can be rebuilt.',
    '对比': 'Contrast', '导出 TIFF 图层包': 'Export TIFF layers', '导出编辑图层': 'Export editing layers',
    '尚无本次结果。': 'No result is available yet.', '尚未选择要保存的变更。': 'Choose changes to save first.',
    '已完成的堆栈任务': 'Completed stack job', '已有银河确认记录': 'Existing Milky Way confirmation',
    '已绑定的无星背景': 'Bound starless background',
    '已读取 {0} 条候选；分类与保留意图分开记录。': 'Loaded {0} candidates. Identity and keep/omit decisions are recorded separately.',
    '已跳过银河增强：': 'Milky Way enhancement skipped: ', '总力度': 'Overall strength',
    '恢复自动': 'Reset to automatic', '成片 TIFF': 'Finished TIFF', '成片处置': 'Image decision',
    '我已确认选区覆盖可见银河': 'I confirm this region contains a visible Milky Way',
    '打开本次结果': 'Open result', '撤销人工身份': 'Clear manual identity',
    '操作未成功完成：': 'Operation failed: ', '无完成状态': 'No completion status',
    '未完成：': 'Not completed: ',
    '未校色输入背景派生 · 另存，不覆盖默认成片': 'Background variant from uncalibrated input · Saved separately',
    '未确认：': 'Not confirmed: ', '正在处理…': 'Processing…', '正在处理高级操作…': 'Processing advanced operation…',
    '此旧完整任务冻结了原作业索引，仅能读取复核；需新快照任务才能追加亮迹派生。': 'This legacy workflow is read-only for review. A workflow with a new stack snapshot is needed to add a trail variant.',
    '此条暂无位置预览': 'No position preview is available.', '渐变与偏色': 'Gradient and color cast',
    '源图或银河蒙版在确认后已改变，请重新查看并确认。': 'The image or Milky Way mask changed. Inspect and confirm the region again.',
    '确认依据不足': 'Insufficient confirmation', '等待设置': 'Waiting for settings', '结构': 'Structure',
    '背景协调': 'Background balance', '背景提亮': 'Background brightness',
    '背景采样蒙版 PNG': 'Background sampling mask (PNG)', '色彩': 'Color',
    '请先读取并选择一条候选。': 'Load and select a candidate first.',
    '请先选择实际源图和银河蒙版。': 'Select the source image and Milky Way mask first.',
    '请手动打开：': 'Open this location manually: ', '读取候选亮迹': 'Load trail candidates',
    '身份：': 'Identity: ', '选择候选后显示原位置参考': 'Select a candidate to see its original position.',
    '选择文件': 'Choose file', '选择文件夹': 'Choose folder', '配对的无亮迹背景': 'Paired background without trails',
    '银河局部增强': 'Local Milky Way enhancement', '银河渐变蒙版': 'Feathered Milky Way mask',
    '需要已有确认记录，或由你明确确认可见银河；尚未运行。': 'Provide a confirmation record or explicitly confirm a visible Milky Way before running.',
    '高级操作已完成；可在相应窗口打开结果。': 'Advanced operation complete. Open the result from its dialog.',
    '高级操作派生 · 基础成片保持': 'Advanced variant · Original result retained',
    '等待确认': 'Awaiting confirmation', '需要复核': 'Review required',
    '明确保留': 'Explicitly keep', '明确省略': 'Explicitly omit',
    '建议保留（未采用）': 'Suggested keep (not applied)',
    '自动建议不是身份认证；未知不等于卫星。保留只加回已测量范围。': 'Suggestions do not certify identity; unknown does not mean satellite. Keep restores only the measured extent.',
    '候选未测量，缺少可靠位置，无法生成邻帧对照': 'This candidate has no reliable measured position; neighboring-exposure views cannot be generated.',
    '序列边界，没有相邻曝光': 'End of sequence: no neighboring exposure.',
    '此曝光未保留已校验的图像，不能冒充已有邻帧证据': 'No verified image was retained for this exposure; it cannot be used as neighboring-frame evidence.',
    '此曝光缺少检查点内容校验，不能当作可靠复核证据': 'This exposure lacks checkpoint verification and cannot be used as reliable review evidence.',
    '实际配准曝光，G 通道、同一显示拉伸；用于检查连续轨迹，不代表最终色彩': 'Actual aligned exposures, green channel with a shared display stretch. Use these to check continuity; they do not show the final color.',
    '实际转换曝光的同一原图区域（未对齐，星空可随时间移动），G 通道、同一显示拉伸；用于检查连续轨迹，不代表最终色彩': 'Same region of actual converted exposures (unaligned; stars may move with time), green channel with a shared stretch. Check continuity here, not final color.',
}

EN.update({'清除地景': 'Clear foreground', '清除银河': 'Clear Milky Way'})

# 拓星 redesign (2026-09-27): overlay colours, legend, discard/save prompts, 底片改判.
EN.update({
    '橙色是地景，橙色实线是它的边界；粉紫双线是银河。点“使用这些分区”后生效。选择“拖出银河椭圆”后，可拖动中心和黄色控制点移动、缩放、旋转；方形黄点调整羽化。Shift 拖动重画，滚轮缩放，中键平移。':
        'Orange marks the foreground, with a solid orange line along its edge; the pink double line marks the Milky Way. Changes apply when you click “Use these regions”. With “Draw Milky Way ellipse” selected, drag the centre or the yellow handles to move, resize and rotate; use the square yellow handle to feather. Shift-drag to redraw, scroll to zoom, middle-drag to pan.',
    '橙色是地景，橙色实线是它的边界；粉紫双线是银河。绘制即生效，无需再次确认。银河可拖动中心和黄色控制点移动、缩放、旋转；方形黄点调整羽化。Shift 拖动重画，滚轮缩放，中键平移。':
        'Orange marks the foreground, with a solid orange line along its edge; the pink double line marks the Milky Way. Drawing activates the region. Drag the centre or the yellow handles to move, resize and rotate; use the square yellow handle to feather. Shift-drag to redraw, scroll to zoom, middle-drag to pan.',
    '绘制工具': 'Drawing tool',
    '地景边界': 'Foreground edge', '银河范围': 'Milky Way extent',
    '建议边缘': 'Suggested edge', '地平线点与手柄': 'Horizon points and handles',
    '分区尚未使用': 'Regions not used yet',
    '本次绘制还没有使用。要使用这些分区，还是放弃更改？': 'This drawing has not been used yet. Use these regions, or discard the changes?',
    '放弃更改': 'Discard changes', '继续编辑': 'Keep editing',
    '不保存': "Don't save", '继续复核': 'Keep reviewing',
    '身份': 'Identity', '处置': 'Decision', '错误': 'Error',
    '本条尚无本次改判。': 'No revised choice for this candidate yet.',
    '人工改判': 'Revised by you', 'Agent 草稿改判': 'Revised in an Agent draft',
    '原值：{0}': 'Previous: {0}',
})

# Trail review fixes (2026-09-27): identity column, filter, identity colours, zoomable
# preview, meteor default keep, trail layer export.
EN.update({
    '选中一条或多条亮迹，分别决定身份与是否保留。测量可靠的流星默认保留（空心圈＝算法）；其他身份默认省略，明确选择“保留”才会加回。改判后列表与右侧预览立即更新，朱点表示你的改判。\n未知不等于卫星；自动省略不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。':
        'Select trails and choose identity and keep/omit decisions separately. Meteors with a reliable measurement are kept by default (hollow ring = algorithm); other identities are omitted unless you choose Keep. The list and the preview update as soon as you revise a trail; a red dot marks your revision.\nUnknown does not mean satellite, and automatic omission does not rule out a meteor. Save a draft, then apply all choices. Preservation covers only the measured extent.',
    '\n测量可靠的流星默认加回；其他身份只有明确选择保留才会加回。':
        '\nMeteors with a reliable measurement are added back by default; other identities only with an explicit Keep.',
    '默认保留（算法）': 'Keep by default (algorithm)',
    '默认省略（测量不可靠）': 'Omit by default (measurement unreliable)',
    '默认保留（按改判身份）': 'Keep by default (revised identity)',
    '明确保留（未测量，不会加回）': 'Explicitly keep (unmeasured; cannot be added back)',
    '恢复默认（按身份）': 'Reset to default (by identity)', '恢复默认': 'Reset to default',
    '人工确定': 'Set by you', 'Agent 确定': 'Set by an Agent', '算法': 'Algorithm',
    '身份：{0} · 来源：{1}': 'Identity: {0} · Source: {1}',
    '处置：{0} · 来源：{1}': 'Decision: {0} · Source: {1}',
    '算法建议：{0}': 'Algorithm suggestion: {0}',
    '筛选': 'Filter', '全部': 'All', '已改判': 'Revised',
    '只显示身份或处置由人工或 Agent 改判过的候选。': 'Show only candidates whose identity or decision was revised by you or an Agent.',
    '显示 {0} / {1} 条': 'Showing {0} of {1}',
    '当前筛选下没有候选。': 'No candidates match this filter.',
    '亮迹位置预览': 'Trail position preview',
    '定位到所选': 'Zoom to selection', '放大': 'Zoom in', '缩小': 'Zoom out',
    '滚轮缩放，拖动空白处或按住中键平移，双击空白处复原；Ctrl 点选可多选。':
        'Scroll to zoom; drag empty space or middle-drag to pan; double-click empty space to reset; Ctrl-click to select several.',
    '会加回': 'Added back', '不加回': 'Not added back', '选中': 'Selected',
    '加回后导出亮迹层': 'Export trail layer after adding back',
    '另存一张只含已保留亮迹的透明 16 位 TIFF 图层（trails_layer_rgba16.tif），叠在基础成片上即得到加回亮迹的成片。至少保留一条亮迹时才导出。':
        'Also save a transparent 16-bit TIFF layer with only the kept trails (trails_layer_rgba16.tif). Placed over the base image, it reproduces the image with trails. It is exported only when at least one trail is kept.',
    # Recall review messages shown under the selected candidate.
    '已请求保留，但未能测量并兑现；请优先检查原片与标记范围。':
        'Keep was requested but could not be measured or applied. Check the original exposure and the marked extent first.',
    '记录为保留；是否实际加回及完整范围仍以应用证据为准。':
        'Recorded as kept. Whether it was actually added back, and over what extent, depends on the application evidence.',
    '已有明确人工标签或省略处置；这记录了取舍，不认证物理身份。':
        'An explicit manual label or omit decision exists. It records a choice; it does not certify the physical identity.',
    '已自动省略，但身份尚未核实，不能视为已排除流星。':
        'Omitted automatically, but the identity has not been verified; this does not rule out a meteor.',
    '候选未能测量而被省略，不能据此确认原片没有流星。':
        'The candidate could not be measured and was omitted; this does not confirm that the original has no meteor.',
})

# Trail review triage (2026-11): priority and keep filters, sortable columns, keys, side panel, composite list.
EN.update({
    '高': 'High', '一般': 'Normal', '低': 'Low', '优先级': 'Priority', '加回': 'Added back',
    '时间': 'Time', '信噪比': 'S/N', '长度': 'Length',
    '点击按帧序号（加入照片的顺序）排列。': 'Click to sort by frame number (the order the photos were added).',
    '先看哪些的提示，不是可信度。自动不加回的候选里，端点渐隐、碰到画框边或测量到边界的为“高”；已请求保留但未能测量的也为“高”。已加回或此前已明确选择的显示“—”；本窗口里的新选择应用后才更新。':
        'A hint for what to look at first, not a confidence. Among candidates not added back automatically, those with a fading end, an end at the frame edge or a measurement at its limit are High; so are keep requests that could not be measured. Candidates already added back or chosen earlier show “—”; choices made in this window update it after they are applied.',
    '复核优先级只是先看哪些的提示，不是可信度。': 'Review priority is a hint for what to look at first, not a confidence.',
    '拍摄时刻，按这台电脑的时区显示。相机和电脑设在同一时区时就是相机上的时间；不同时会差整小时。':
        'Capture time in this computer’s time zone. It matches the camera clock when the camera and this computer use the same zone; otherwise it is off by whole hours.',
    '{0}（这台电脑的时区 {1}）\n记录的世界时：{2}': '{0} (this computer’s time zone, {1})\nRecorded UT: {2}',
    '沿轨迹的总信噪比（S/N），可当亮度参考：越大越明显；轨迹越长也越大，不等于亮度。':
        'Total S/N along the trail, a rough guide to brightness: higher means more distinct; it also grows with length, so it is not brightness.',
    '测量出的长度（像素）。': 'Measured length in pixels.',
    '下一条待复核': 'Next to review',
    '跳到列表中下一条还没有选择的候选（回车）。': 'Go to the next candidate in the list without a choice yet (Enter).',
    'K 保留': 'K keep', 'O 不加回': 'O omit', 'U 恢复默认': 'U reset', '回车 下一条': 'Enter next',
    '不加回（选择）': 'Omit',
    '快捷键作用于选中的候选；焦点在列表或预览上时可用。':
        'Shortcuts act on the selected candidates while the list or the preview has focus.',
    '先选择候选。': 'Select candidates first.',
    '当前筛选下的候选都已有选择。': 'Every candidate in this filter already has a choice.',
    '旁边已有同名的 {0}，导出时会一起替换。替换吗？': '{0} already exists beside it and would be replaced too. Replace it?',
    '替换': 'Replace',
    '详情': 'Details',
    '显示或收起右侧的缩略图、邻帧对照与改判记录。':
        'Show or hide the side panel with the thumbnail, neighbouring exposures and revisions.',
    '扫描缩略图': 'Scan thumbnail',
    '左：本张曝光；右：减去堆栈结果后剩下的部分。': 'Left: this exposure. Right: what remains after subtracting the stacked image.',
    '此候选没有扫描缩略图。': 'This candidate has no scan thumbnail.',
    '扫描缩略图无法读取。': 'The scan thumbnail cannot be read.',
    '导出合成清单…': 'Export composite list…', '导出合成清单': 'Export composite list',
    '合成清单': 'composite-list', 'CSV 表格 (*.csv)': 'CSV table (*.csv)',
    '把含亮迹版本加回的每条流星（来自哪张原片、拍摄时刻、位置）存成 CSV 和 JSON，保存在你选的位置。':
        'Save each meteor added back in the version with trails (source photo, capture time, position) as CSV and JSON, in a place you choose.',
    '这次复核还没有对应的含亮迹版本，暂时不能导出合成清单。':
        'There is no version with trails for this review yet, so no composite list can be exported.',
    '作品文件夹只放成片图像，请把清单存到别处。': 'The works folder holds only the delivered images. Save the list somewhere else.',
    '已导出合成清单：加回 {0} 条。\n{1}\n{2}': 'Composite list exported: {0} added back.\n{1}\n{2}',
    '\n清单按已出图的版本；本窗口里还没应用的选择不在其中。':
        '\nThe list describes the delivered version; choices not yet applied in this window are not in it.',
    '没有与当前候选记录对应的含亮迹版本，不能导出合成清单。':
        'No version with trails matches the current candidate record, so no composite list can be exported.',
    '加回记录里的候选不在当前复核数据中，不能导出合成清单。':
        'A candidate in the add-back record is missing from the review data, so no composite list can be exported.',
    '拍摄时刻：capture_time_utc 是世界时，capture_time_local 按这台电脑的时区换算。相机 RAW 的世界时由 Siril 按处理时电脑的时区从相机时间换算，所以相机和电脑设在同一时区时，本地时刻就是相机上的时间；相机设在别的时区时会差整小时。':
        'Capture times: capture_time_utc is UT, and capture_time_local converts it to this computer’s time zone. For camera RAW, Siril derives UT from the camera clock with the processing computer’s time zone, so the local time is the camera clock when the camera and the computer use the same zone; with the camera set to another zone it is off by whole hours.',
    '含亮迹版本里的线性母版（master_scene_linear.tif）不含流星；流星只加在成片（final_srgb16.tif、final.jpg）里。':
        'The linear master in the version with trails (master_scene_linear.tif) contains no meteors; meteors are added only to the photographs (final_srgb16.tif, final.jpg).',
    '身份是算法建议或复核者的选择，不是认证；同一张曝光的一颗流星可能有多条候选。':
        'Identities are algorithm suggestions or reviewer choices, not certification; one meteor in one exposure may appear as several candidates.',
    '未扫描的帧（质量门剔除或未能对齐）里即使有流星，也不在本清单中。':
        'Frames that were not scanned (excluded by the quality gate or not aligned) are not covered, even if they hold meteors.',
    'exposure_s 是每行那张曝光的时长（秒，取自转换后帧的 FITS EXPTIME），没有记录时留空；exposure 是本段合计。':
        'exposure_s is the exposure time of each row’s exposure in seconds (FITS EXPTIME of its converted frame), empty when it was not recorded; exposure gives the segment totals.',
    '若加入照片时相机 RAW 被转成了 DNG 工作副本，清单里写的是工作副本的文件名：原片文件名主干相同，哈希不同（没有找到与之相符的转换记录）。':
        'If camera RAW files were converted to DNG working copies when they were added, the list names the working copies: each original has the same name stem and a different hash (no matching conversion record was found).',
})

# Keep or omit everything the current filter shows (2026-11).
EN.update({
    '当前筛选全部…': 'Everything in filter…',
    '对当前筛选显示的全部候选一次设为保留或不加回；先选一个筛选。':
        'Keep or omit every candidate the current filter shows. Choose a filter first.',
    '全部保留…': 'Keep all…', '全部不加回…': 'Omit all…',
    '全部保留': 'Keep all', '全部不加回': 'Omit all', '整批设置': 'Set in one step',
    '先选一个筛选，再对筛选结果整批设置。': 'Choose a filter first, then set everything it shows.',
    '已把当前筛选的 {0} 条设为“{1}”；“撤销上一步”可恢复。':
        'Set {0} candidates in this filter to “{1}”. Undo last change restores them.',
    '把当前筛选的 {0} 条全部设为保留？\n没看过的候选里可能有飞机或卫星，保留后会加进含亮迹版本。“撤销上一步”可恢复。':
        'Keep all {0} candidates in this filter?\nCandidates you have not looked at may be aircraft or satellites; kept ones are added to the version with trails. Undo last change restores them.',
    '把当前筛选的 {0} 条全部设为不加回？\n“撤销上一步”可恢复。':
        'Omit all {0} candidates in this filter?\nUndo last change restores them.',
    '\n另有 {0} 条刚改过、已不在此筛选，不会改动。':
        '\n{0} more you just changed no longer match this filter and stay as they are.',
})

# Plainer review wording (2026-11): 不加回 for 省略, 是否加回 for 处置. English keeps Keep / Omit for the
# choices and says Added back / Not added back for the filters and the legend.
EN.update({
    '选中一条或多条亮迹，分别决定身份与是否保留。测量可靠的流星默认保留（空心圈＝算法）；其他身份默认不加回，明确选择“保留”才会加回。改判后列表与右侧预览立即更新，朱点表示你的改判。\n未知不等于卫星；自动不加回不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。':
        'Select trails and choose identity and keep/omit decisions separately. Meteors with a reliable measurement are kept by default (hollow ring = algorithm); other identities are omitted unless you choose Keep. The list and the preview update as soon as you revise a trail; a red dot marks your revision.\nUnknown does not mean satellite, and automatic omission does not rule out a meteor. Save a draft, then apply all choices. Preservation covers only the measured extent.',
    '选中一条或多条亮迹，分别决定身份与是否保留。只有明确选择“保留”才会加回，判断为流星也不会自动加回。\n未知不等于卫星；自动不加回不代表已排除流星。先保存草稿，最后一次应用全部选择；保留限于已测量范围。':
        'Select trails and choose identity and keep/omit decisions separately. Only Keep adds a trail back; a Meteor label alone does not.\nUnknown does not mean satellite, and automatic omission does not rule out a meteor. Save a draft, then apply all choices. Preservation covers only the measured extent.',
    '\n未能测量定位，不能把“不加回”视为已排除流星':
        '\nPosition could not be measured. Omission does not rule out a meteor.',
    '恢复默认（不加回）':
        'Reset to default omission',
    '是否加回：保持原样':
        'Decision: unchanged',
    '是否加回':
        'Decision',
    '只显示身份或是否加回由人工或 Agent 改判过的候选。':
        'Show only candidates whose identity or decision was revised by you or an Agent.',
    '先选择候选和要更改的身份或是否加回。':
        'Select candidates, then choose an identity or a keep/omit decision.',
    '是否加回：{0} · 来源：{1}':
        'Decision: {0} · Source: {1}',
    '{0} · {1}\n已记录身份：{2}；是否加回：{3}。 ':
        '{0} · {1}\nRecorded identity: {2}; review decision: {3}. ',
    '曝光 {0} 秒':
        'exposure {0} s',
    '默认不加回（测量不可靠）':
        'Omit by default (measurement unreliable)',
    '明确不加回':
        'Explicitly omit',
    '默认不加回':
        'Omit by default',
})

# The trail scan writes its suggestion reasons in English (transient_trails.suggest). The Chinese interface
# shows them in Chinese through these display-only patterns: (regular expression, Chinese template). A
# reason that matches none is shown as written; the English interface always shows the original.
ZH_SUGGESTION_REASONS = (
    (r"not measured \((?P<status>[^)]*)\)", "未能测量（{status}）"),
    (r"same place in (?P<count>\d+) frames: (?P<frames>.*)", "在 {count} 张曝光的同一位置都有：{frames}"),
    (r"periodic brightness modulation along the trail \(period (?P<period>[^ ]+) px, "
     r"autocorrelation (?P<correlation>[^,]+), dark fraction (?P<dark>[^)]+)\)",
     "沿轨迹亮度周期性明暗（周期 {period} 像素，自相关 {correlation}，暗段占 {dark}）"),
    (r"geometrically compatible segment in adjacent frame (?P<frame>\S+) \(angle (?P<angle>[^°]+)°, "
     r"lateral (?P<lateral>\S+) px, gap (?P<gap>\S+) px\); same-object identity unverified",
     "相邻曝光{frame}里有一段方向和位置都接得上（夹角 {angle}°，横向偏 {lateral} 像素，相隔 {gap} 像素）；是否同一物体未核实"),
    (r"total S/N (?P<snr>\S+) below (?P<weak>\S+)", "总信噪比 {snr}，低于 {weak}，太弱，不判断身份"),
    (r"single frame candidate; no compatible continuation was linked in the scanned adjacent-frame catalog "
     r"\((?P<frames>.*)\); original-exposure absence is not established; no periodic modulation, "
     r"total S/N (?P<snr>\S+)",
     "只在本张出现：已扫描的相邻曝光（{frames}）里没有接得上的一段，但不能断定原片里没有；没有周期性明暗，总信噪比 {snr}"),
    (r"light curve: start (?P<start>\w+)(?: \((?P<start_level>[\d.]+) of the peak\))?, "
     r"stop (?P<stop>\w+)(?: \((?P<stop_level>[\d.]+) of the peak\))?",
     "亮度变化：起点{start}{start_level}，终点{stop}{stop_level}"),
    (r"both measured ends fade inside the frame; supports a meteor suggestion, not physical identity certification",
     "两端都在画面内渐隐：支持“流星”的建议，但不是身份认证"),
    (r"both measured ends stop abruptly near full brightness; compatible with a shutter-limited trail, "
     r"not physical identity certification",
     "两端都在接近最亮处突然中断：与被快门截断的轨迹相符（常见于卫星），但不是身份认证"),
    (r"the ends do not decide identity; an abrupt or unjudged end does not exclude a meteor "
     r"\(unjudged: too faint, no data beyond, or at the frame border\)",
     "两端不能决定身份：一端突然中断或无法判断，都不排除流星（无法判断：太暗、外侧没有数据或碰到画框边）"),
)
# Values inside those reasons, and how levels and frame names are shown.
ZH_REASON_VALUES = {
    "faded": "渐隐", "abrupt": "突然中断", "between": "介于渐隐和中断之间", "unjudged": "无法判断",
    "too_short": "太短", "no_signal": "没有足够信号", "none available": "无",
}
ZH_REASON_LEVEL = "（峰值的 {value}）"
ZH_REASON_FRAME = "第{number}张"
ZH_REASON_LIST_SEPARATOR = "、"

# Engine record words shown in the Chinese interface (display only; records keep their words): older
# records and the recall review say 省略 where the dialog says 不加回.
ZH_PLAIN_RECORD_WORDS = {
    '默认省略': '默认不加回',
    '明确省略': '明确不加回',
    '已自动省略，但身份尚未核实，不能视为已排除流星。': '已自动不加回，但身份尚未核实，不能视为已排除流星。',
    '候选未能测量而被省略，不能据此确认原片没有流星。': '候选未能测量，没有加回；不能据此确认原片没有流星。',
    '已有明确人工标签或省略处置；这记录了取舍，不认证物理身份。': '已有明确人工标签或不加回的选择；这记录了取舍，不认证物理身份。',
}

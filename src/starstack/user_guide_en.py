"""Complete offline English help; anchors match the Chinese guide."""

HELP_MARKDOWN_EN = '''# Aester · Start creating

## Installation and first launch

Run the Aester installer and choose an installation folder. It prepares the basic runtime and Siril during installation. An internet connection is needed at this stage; if a download stops, retry to reuse files that have already passed verification. For the Python packages, the installer first times the official servers and the listed mirrors in mainland China and downloads from the fastest; Siril comes only from its official site, free-astro.org, with no mirror. Every file is checked against its SHA-256.

**Reduce stars** and starless layers need the nox star-removal model. The rights to this third-party model's training data are still being checked, so it is not offered for download and these features are unavailable for now. If you already have the conversion verified for this project, select it under **Settings → Components & prompts**. After changing the runtime, save your editing project and restart when prompted.

The installer is not code-signed. After downloading, you can compare it with `SHA256SUMS.txt` next to the installer on the GitHub Releases page: open PowerShell in the folder that holds the installer, run `Get-FileHash <installer file name> -Algorithm SHA256`, and check that the Hash equals the line for the same file (upper or lower case does not matter). With Smart App Control on in Windows 11, Aester may be blocked; see “Blocked by Smart App Control” below.

On the welcome screen, choose **Start using Aester**, **Meteor shower** or **Connect an AI assistant**. No language model is needed for any installed local processing feature. You can reopen the welcome screen from Help.

Under **Settings → Appearance**, choose a light or dark theme. The choice is remembered. The same menu also holds **Night vision** and **Image surround**; see “Viewing conditions and interface marks” below. Light and dark themes change the interface only: photo previews, region overlays and finished images keep their original colours.

Under **Settings → Language**, choose **English** or **中文**. Save your editing project and restart Aester to apply the language throughout the interface. The choice is remembered. Error messages follow the interface language as well. Language changes do not alter your photographs, processing recipes, file paths or MCP protocol keys.

<a name="sac"></a>

## Blocked by Smart App Control

When Smart App Control is on in Windows 11, it blocks programs that Microsoft cannot judge to be safe and that have no valid signature, and it cannot make an exception for a single program. Aester and the official Siril it uses are not code-signed, so they may be blocked.

To check whether it is on, open **Windows Security → App & browser control → Smart App Control settings**. The state is On, Evaluation or Off; in Evaluation it blocks nothing.

If Aester is blocked, the only remedy is to set it to Off in the same place. Before you do, note that whether you can turn it on again depends on your Windows version; on older versions it can only be turned on again after resetting or reinstalling Windows. Windows 10 does not have this feature.

<a name="start"></a>

## Your first photo in four steps

The stage bar across the top of the window shows 01 Photos, 02 First frame, 03 Regions, 04 Stack and 05 Photograph; **Let an Agent retouch** at its end is 06 Refine. The stage bar shows progress and offers shortcuts; it never switches pages. Available stages can be clicked, and hovering over an unavailable stage explains why.

The viewing area is on the left. The right column follows the working order: ① Regions, ② Colour and ③ Optional effects, with **Save photographs to** and **Start stacking / Stop** fixed at its bottom. The readout strip along the bottom of the window shows the stage, progress, percentage and the full message; **Details** opens the processing record.

1. **Add photos.** Select a sequence of RAW exposures taken from the same position with the same composition. You can select multiple files, drag them into the window, use **Add photos** and **Remove selected** above the viewing area, or use **Clear all** to empty the selection without deleting originals. At least three are needed for stacking and at most 300 per job; a single file can still be previewed. The first photo in the list below the image (the first frame) is the colour reference. Each small dot above the image is one photo; the ringed dot is the first frame.
2. **Mark relevant regions.** Choose **Mark sky / foreground / Milky Way** in ① Regions on the right. Draw the foreground boundary if present, and an ellipse over a visible Milky Way. An unmarked foreground means the entire image is sky; an undrawn Milky Way stays off. No repeated confirmation is needed.
3. **Adjust colour and choose effects.** The sliders in ② Colour change warmth, green–magenta tint and exposure. All three effects in ③ Optional effects start off (once meteor-shower mode is on, **Preserve meteors and trails** is still ticked the next time you start Aester). Colour dragging uses a small responsive proxy; a sharper preview follows when you stop.
4. **Start stacking.** Choose **Start stacking** at the bottom of the right column; the line above it shows the cache space the job needs. When processing finishes, choose **Open photographs folder** in the view row (also in the File menu) to find your linear master, 16-bit finished TIFF and JPEG.

Processing happens on your computer and leaves the original files unchanged. Start with a small set. Do not mix different nights, changed compositions or substantially different exposures in one job. Split a sequence of more than 300 photos into segments; each segment gives its own image, and segments are not combined automatically. The longest continuous sequence tested so far is about 31 minutes; until longer ones are tested, keep each segment to about 30 minutes and no more than about 150 photos, whichever comes first. For a meteor shower, “[Meteor showers](#meteor-shower)” below goes through the whole night step by step.

<a name="colors"></a>

## Correcting photo orientation

New selections follow the capture orientation recorded in the first photo, using the same direction for the whole group. Below the image, use **Rotate left 90°** or **Rotate right 90°** to turn the whole group, or **Reset orientation** to return to the capture orientation, for RAW or DNG photos. Originals remain unchanged. Rotation only reorders pixels; it does not crop or interpolate them.

Set the direction before drawing foreground or Milky Way regions. Changing it clears existing marks and asks you to redraw. Stacking waits for the updated reference. The direction is saved with the editing project and used by the master, TIFF and JPEG. Older projects retain their existing direction and masks.

## Adjusting colour

In ② Colour, move Warmth to the right for a warmer image, and Tint to the right for more magenta. Exposure is measured in EV; positive values brighten the image. You can also type a value into the field beside each slider; it applies when you press Enter or leave the field, and exposure moves in 0.1 EV steps. The mouse wheel scrolls the panel and never changes a slider or number box. Double-click a slider to reset it to 0. **Reset adjustments** in the section heading restores the captured white balance and default tone settings.

Warmth and tint are relative adjustments, not measured colour temperatures. The first-frame preview and final photo share the colour-processing logic and use the first selected photo as the colour reference. Sky and foreground receive the same colour adjustment.

The **Camera look** label in the view row shows the look used by the first-frame preview: the camera body, the borrowed look table and its strength. When no look applies, it reads “Camera look · none”. A look affects only the colour style of the preview and finished photo; originals are unchanged.

Stacking clears the first-frame preview. To see it again, click **02 First frame** in the stage bar or **Refresh first-frame preview** in the view row; it is regenerated from the current photos and colour settings.

A single exposure is noisier than a stack, and a reduced preview cannot show every star. The preview and final photo therefore do not match pixel for pixel. Deep-sky FITS material currently uses basic rendering and does not support these camera colour sliders; this is not camera RAW colour calibration.

- **JPEG:** ready to view and share.
- **16-bit finished TIFF:** already has display tones applied and can be edited in familiar photo software.
- **Float32 linear master TIFF:** preserves more room for further processing, including negative values and highlight headroom. A dark appearance in a normal viewer is expected. Use a workflow that supports linear images. The appearance sliders do not overwrite this master.

<a name="viewing"></a>

## Viewing conditions and interface marks

**Image surround.** Under **Settings → Appearance → Image surround**, choose **Dark grey · default**, **Mid-grey evaluation** or **Black · viewing only**. The **Grey** button below the image switches mid-grey evaluation directly: the area around the photo turns mid grey so you can judge brightness and contrast before exporting. A black surround makes the photo look brighter and suits viewing only. The surround changes the display, never the photograph.

**Night vision.** To protect dark adaptation in the field, turn it on from **Settings → Appearance → Night vision**, the **Night** button below the image or Ctrl+Alt+N, and use the same control to turn it off. The whole interface turns pure red, the image is shown as a single red channel, and a **Not for colour judgement** label stays in its top-left corner. Night vision changes the display only; do not use it to judge colour. Exported photographs are unaffected.

**Who decided.** Small dots show who made a decision. A filled vermilion dot means you decided, a filled pale-gold dot means an Agent was involved, and a hollow ring marks an algorithmic suggestion that nobody has confirmed. After you save regions in the region editor, a vermilion dot and the seal 人定 (“decided by a person”) appear beside the ① Regions heading; the seal is the same in both interface languages. Regions drawn by an Agent through MCP show only a pale-gold dot, never the seal. The pale-gold dot beside **Let an Agent retouch** marks a step that an external Agent carries out, and trail review marks “Revised by you” and “Revised in an Agent draft” separately.

**Activity mark.** While Aester is processing, a brightness sweep runs over 28 fixed lines (the twenty-eight lunar lodges) at the left of the readout strip, on the current stage ring (04 while a job runs, 02 while the first frame is being built) and on the empty viewing area. It is drawn still when Windows animation effects are off, pauses while the window is minimised, and turns red in night vision.

Error messages appear in the readout strip in vermilion text, accompanied by an error icon.

<a name="masks"></a>

## Sky and foreground

Stars move between exposures while the landscape is usually stationary, so the two areas need separate alignment. Open **Mark sky / foreground / Milky Way** in ① Regions on the right:

1. Click a few points along the ridge from left to right, then choose **Finish horizon**. Orange marks the foreground, and a solid orange line is its boundary. Yellow dots are horizon points and handles.
2. Use **Paint foreground** for roofs, trees and equipment. Use **Restore sky** to correct an oversized selection. Adjust the brush size, zoom with the mouse wheel and pan with the middle mouse button. **Fit to window** returns to the whole image.
3. **Find nearby edge** proposes a dashed light-blue line, and **Accept light-blue edge** is highlighted in the same light blue while the suggestion waits. Once the line follows the real horizon, choose that button, or keep your original line. This looks for local changes in brightness; it cannot know which edge is a mountain ridge.
4. Choose **Use these regions**. Drawing takes effect directly: Aester saves the full-resolution regions for you, and no separate mask file needs to be prepared.

Undo and redo let you revise the drawing. Existing external greyscale PNG masks can still be loaded from Settings. Use **Clear foreground** to return to an all-sky image. The brush circle shows its actual coverage and follows zoom. If a drawing has not been used yet, Cancel, Esc or closing the window first asks whether to use these regions, discard changes or keep editing. Changing the selected photos or processing backend requires the previous regions to be confirmed again, so a drawing is not silently applied to a different image.

Draw a rough outline first, then zoom in to inspect its edges. A static region cannot repair moving leaves, intricate gaps between branches or large moving clouds. Gentle edge blending cannot guarantee a clean join in these situations.

<a name="stars"></a>

## Reduce stars

Enable **Reduce stars** in ③ Optional effects when the star field looks too crowded. The software's recommendation is selected initially; you can turn off automatic advice and choose a strength from 0–100%.

At 0%, the image is unchanged. Intermediate settings mainly reduce faint and medium-bright stars. High strengths also weaken bright stars; 100% removes the entire star layer separated by the current backend. The recommendation is an adjustable aesthetic starting point. The cleanliness of the starless background still depends on that backend, and halos may remain around bright stars.

A star removal model must be configured. The nox model is not offered for download for now, so without the conversion verified by this project this effect cannot be used. Installing or selecting a model does not enable this effect automatically. Basic stacking still works without a model.

After drawing the Milky Way, you can set its relative offset. For example, background 80% with offset −20% gives 60% in the Milky Way. A negative offset leaves more stars inside it. The offset takes effect only for a drawn Milky Way. When star reduction is enabled, the displayed and delivered final photo is the reduced version.

For layered editing, also tick **Also save starless image** or **Also save star layers**. The first saves `starless_srgb16.tif` in the photographs folder; the second saves all separated stars as `stars_add_srgb16.tif` (Add / Linear Dodge) and `stars_subtract_srgb16.tif` (Subtract). Starless − subtract layer + add layer = the photograph before star reduction, whatever the strength setting. Stars are removed only in pure sky; the foreground and the feathered horizon band are unchanged. The star-removal model is experimental, so small dark or coloured rings and halos may be visible in the starless image. Blend the star layers with Add and Subtract, not Screen.

<a name="meteor-shower"></a>

## Meteor showers

This section follows one night of meteor-shower photos in order: each segment becomes one photograph with its meteors added back, and then you check the candidates. Each control is described in more detail in “[Keeping meteors and other bright trails](#trails)” below. The section describes versions after v1.0.0-rc.1; rc.1 does not have the **Meteor shower** button, the space estimate before start, review priorities or the composite list.

### Shooting and segments

- Keep the camera on a tripod in one position with one composition, and shoot RAW continuously. Keep exposure, ISO and white balance fixed.
- Shoot continuously, and add every frame of a segment; do not pick every second or third one. To tell satellites and aircraft from meteors, Aester looks at whether a trail continues in the exposure before or after it. In a test with photos from one night, 9 trails were added back automatically from every third frame. With all frames, the algorithm reclassified 3 of them as satellites because they continued in a neighbouring exposure, and 2 more became unknown and were no longer added back either.
- A job takes at most 300 photos. The longest continuous sequence tested so far is about 31 minutes; until longer ones are tested, keep each segment to about 30 minutes and no more than about 150 photos, whichever comes first.
- Each segment gives its own photograph, and segments are not combined. When we merged two segments by hand, the foreground lined up but parts of the sky were 10–25 pixels apart, so we recommend one photograph per segment.
- Nikon High Efficiency RAW (HE, HE★) cannot be read directly. First install Adobe DNG Converter yourself, from **Get the free converter from Adobe** under **Settings → Processing backends**. When you add the photos, Aester converts them to DNG working copies in the cache and leaves the originals untouched. Conversion takes extra time, and each working copy of a 45 MP photo takes about 50 MB.

### Processing one segment

1. **Turn on meteor-shower mode.** Click **Meteor shower** in ③ Optional effects, or the button of the same name on the welcome screen. It ticks **Preserve meteors and trails**, the button reads **Meteor shower · on**, and the mode is still on the next time you start Aester. It does exactly what ticking **Preserve meteors and trails** by hand does and changes no other setting; unticking that box turns the mode off. A few hints appear under the box: mark the foreground in ① first (shown until you have marked it), keep segments to about 30 minutes or about 150 photos, click **Clear all** before the next segment, and clean the cache only after the trail review.
2. **Add the segment's photos** with **Add photos** or by dragging them into the window. If the list would go over 300 photos, that batch is not added and Aester asks you to split the sequence into segments.
3. **Mark the foreground** with **Mark sky / foreground / Milky Way** in ① Regions (see “Sky and foreground”). From the second segment on, if the tripod has not moved, click **Reuse the previous segment’s foreground** below it: the boundary drawn for the previous segment is placed on this segment's first frame. Check it, then click **Use these regions**. The Milky Way is not carried over and has to be drawn again. The button appears once the first-frame preview is ready, if the previous segment's foreground was drawn in Aester and that segment was started or opened in this window. It cannot be used when the photos differ in size or orientation. After restarting Aester, open the previous segment first from **File → Recent jobs**, then click **Clear all** and add the new photos.
4. **Check the space estimate.** Above **Start stacking**, a line reads like “Start needs ~X GB free · cache drive has Y GB”. It uses the same calculation as the space check at start, including a safety margin; hover over it for the breakdown. With default settings, a 45 MP photo needs about 1 GB, so 150 photos need about 150 GB. If the line turns red with “Not enough space: at most N photos per segment”, choose no more photos than that per segment, review and clean the previous segment's cache first, or move the cache to a larger drive under **Settings → Photographs and cache** (this applies to new jobs). “Not enough space: this drive cannot hold one segment” leaves only cleaning or another drive. With the Aester modified build or **Preserve stacking highlights**, a long sequence also gets a suggestion to switch to the official Siril build or to turn highlight protection off: both need much more free space at start, and the tooltip gives the factor.
5. **Start stacking.** The readout strip at the bottom of the window shows the stage and the time elapsed, for example “Finding meteors and trails · elapsed 12:05”; there is no remaining-time estimate. Processing runs in a separate process, so closing the window does not cancel it, and the next segment's photos can be added only after this job has finished. For processing time there are only simulated figures so far: on a workstation limited to 8 cores and 16 GB of memory, 45 MP photos took about 5 seconds each plus about 75 seconds, which works out to about 14 minutes for a 150-photo segment. This does not include converting Nikon High Efficiency RAW to DNG when the photos are added. Disk and single-core speed were not limited in the simulation, so a real laptop will be slower; treat this as a lower bound. No real laptop has been measured yet.
6. **Two photographs.** When the job is done, the result tabs read **With trails / Base photograph** with the with-trails version first, and the note reads like “With-trails version · N added back automatically · H high priority to check (M unverified in all)”. **Open photographs folder** opens the with-trails folder `delivery-trails-<n>kept-<token>`, where n is the number of trails added back; the base folder `delivery-<token>` is next to it. Meteors with a reliable measurement are added back automatically; this is the algorithm's suggestion, not proof of what a trail is. The base photograph is always kept. The linear master in the with-trails folder contains no meteors; they are added to `final_srgb16.tif` and `final.jpg` only.

### Reviewing

Choose **Review trail candidates** at the bottom of the right column. A segment of about 150 photos can have a thousand or more candidates (one 144-photo segment had 1,275), so you do not need to look at every one.

- The list opens sorted by **Priority**, High first. Priority is a hint for what to look at first, not a confidence. A good start is to choose **Added back** in the **Added back** filter, next to **Priority**, and check that each trail added back is a meteor; then set **Added back** back to **All** and choose **High** under **Priority**. The filters apply together, and trails added back automatically have no priority, so choosing both shows nothing.
- Above the list you can filter by identity, **Revised**, **Priority** and **Added back**; each button shows its count. Click a column heading to sort by frame number, priority, time, S/N or length.
- With a candidate selected, press K to keep it, O to omit it or U to reset it to the default; Enter or **Next to review** moves to the next candidate without a choice. The keys work while the list or the image has focus.
- **Details** shows or hides the side panel: the scan thumbnail, the candidate's file name and exposure time, its previous, own and next exposures, and the algorithm's reasons. The scan thumbnail was made during the scan and was not checked for later changes, so it is for browsing only; judge from the verified exposures below it. A trail that continues into the previous or next exposure is usually a satellite or an aircraft.
- The **Time** column is the capture time in this computer's time zone, converted from the UT that Siril recorded. It matches the camera clock when the camera and the computer use the same time zone; otherwise it is off by whole hours.
- With a filter chosen, **Keep all…** or **Omit all…** under **Everything in filter…** sets every candidate the filter shows in one step, after asking for confirmation. Candidates you have not looked at may be aircraft or satellites. **Undo last change** reverses it.
- **Save draft** if you want to continue later; closing the window with unsaved choices asks first. When you are done, **Apply choices and render** saves a new with-trails photograph and keeps the earlier one.

### Exporting the composite list

In the review window, **Export composite list…** writes a CSV file and a JSON file with the same name to a place you choose.

- There is one row for each meteor added back: the original file's name and SHA-256, frame number, capture time in UT and in local time, exposure time, both ends of the trail in the photograph and in its own exposure, and the identity with who set it. The header names the frame the sky was aligned to, the earliest and latest capture times among the candidates, the total exposure, and the photos that were not scanned.
- Files are named by file name only, without drive or user folders.
- The list cannot be saved in the photographs folder; the save dialog starts in Documents.
- The list describes the with-trails version that has been rendered, so choices not yet applied are not in it. **Apply choices and render** closes the review window; once the new render has finished, open **Review trail candidates** again to export the list for the new version. Until a with-trails version exists, the button is unavailable.
- For Nikon HE RAW converted to DNG, a row names the original NEF and its hash, also after a restart, as long as the conversion record in the cache is still there and matches the DNG that was stacked. Without a matching record, the row names the DNG, and the header and the status line say so.
- Photos rejected by the quality check or not aligned were not scanned, so any meteors in them are not in the list; the header lists those photos.

### Coming back to a job

- **File → Recent jobs** lists the last 10 jobs started or opened in the Aester window, with their start time, photo count and whether **Preserve meteors and trails** was on. A job that has been moved or deleted, or whose drive is not connected, is reported and stays in the list; **Clear list** only empties the list and deletes no files.
- **Settings → Photographs and cache** lists the jobs in the cache: select one and click **Open this job**, or double-click it.
- A job in a cache folder you used earlier and have since changed is not listed there; choose **File → Open existing job…** and pick its `workflow.json`.

Once the job is open, **Review trail candidates** works as long as its intermediate files are still there.

### Cleaning up, then the next segment

1. After reviewing and exporting the list, select the job under **Settings → Photographs and cache** and click **Clean intermediate files of selected jobs…**. The photographs stay, but the job's trail candidates can no longer be reviewed and no composite list can be exported for it.
2. For the next segment, click **Clear all**, add its photos and continue from step 3 of “Processing one segment”. Meteor-shower mode stays on.
3. When space is tight, finish each segment (process, review, export, clean) before starting the next one.

### Not tested yet

Continuous sequences longer than about 31 minutes and processing time on a real laptop have not been tested; combining segments into one photograph is not supported (see “Shooting and segments”). The 31-minute figure comes only from a Nikon Z8 sequence at 20 mm with 10-second exposures. In one Sony test at 16 mm with 30-second exposures over about 42 minutes, photos taken long before or after the frame the sky was aligned to could not be aligned or were rejected by the quality check, so only 13 of 78 were used, and those left out were not scanned for meteors. The composite list's header records how many photos were stacked and which were not scanned.

Detection is not guaranteed: a trail added back automatically may be a satellite or an aircraft, and meteors can be missed. To tell us how a night went, choose the **流星雨出图回执 / Meteor shower result** form under Issues on GitHub; please do not upload RAW files to a public issue.

<a name="trails"></a>

## Keeping meteors and other bright trails

Enable **Preserve meteors and trails** in ③ Optional effects. For a meteor shower you can click **Meteor shower** just above it instead, which ticks the box and remembers it; “[Meteor showers](#meteor-shower)” above goes through a whole night. Leave this effect off when it is not needed. Aester then skips the trail scan and does not produce an additional trail-replay version.

When enabled, a new job first finishes the base photograph, then automatically adds back meteor candidates whose measurement is reliable (measured, with no width, profile or sideband flags) and saves them as a separate with-trails version. This is an algorithmic suggestion, not identity certification; other candidates wait for your review, and the base photograph is always kept. The linear master of the with-trails version contains no meteors; they are added to the 16-bit finished TIFF and the JPEG only. Review uses this job's cache, so do not clean it before you have finished reviewing. The photographs folder then holds the base folder `delivery-<token>` and the with-trails folder `delivery-trails-<n>kept-<token>`. The result tabs read **With trails / Base photograph**, with the with-trails version shown first; the note reads like “With-trails version · N added back automatically · H high priority to check (M unverified in all)”, or “N added back (algorithm a / human h / Agent g)” when you or an Agent chose to add back any trails.

To check or change the choices, choose **Review trail candidates**, below **Start stacking / Stop** at the bottom of the right column. After a restart, reopen the job from **File → Recent jobs**, or with **Open this job** under **Settings → Photographs and cache**. In the review window:

- **List and filters.** The columns are **Candidate / exposure**, **Priority**, **Identity**, **Decision**, **Time**, **S/N**, **Length** and **Your choice**. Click a heading to sort by frame number, priority, time, S/N or length. The list opens sorted by priority, High first. Above it, filter by identity (**All**, **Meteor**, **Satellite**, **Aircraft**, **Artifact**, **Unknown**), by **Revised**, by **Priority** (**High**, **Normal**, **Low**) and by **Added back** or **Not added back**; each button shows its count. Priority is a hint for what to look at first, not a confidence: among the candidates not added back automatically, those with a fading end, an end at the frame edge or a measurement at its limit are High.
- **Sources.** The **Identity** and **Decision** columns carry a source dot: filled vermilion for you, pale gold for an Agent, a hollow ring for the algorithm.
- **Whole image.** Select a candidate in the list, or click it on the whole image; Ctrl-click adds to the selection. A selected trail is ringed in yellow; hovering only highlights. Meteors are green, satellites light blue, aircraft orange, artifacts pink-purple and unknown trails white. A solid line will be added back; a dotted line will not. Zoom with the mouse wheel and pan by dragging or with the middle button. Double-click empty space, choose **Fit to window** or press 0 to reset the view; **Zoom to selection** frames the selected trails.
- **Side panel.** **Details** shows or hides it. From the top: the scan thumbnail; the selected candidate's file name, exposure time and recorded identity; your revision of it; the **Previous exposure**, **Source exposure** and **Next exposure**; and the algorithm's suggestion with its reasons. The scan thumbnail was made during the scan and was not checked for later changes, so it is for browsing only; judge from the verified exposures. If reliable neighbours are unavailable, this is stated explicitly.
- **Time.** The **Time** column is the capture time in this computer's time zone, converted from the UT that Siril recorded; hover over it for the date, the zone and the UT. It matches the camera clock when the camera and the computer use the same time zone; otherwise it is off by whole hours.
- **Choices.** Identity (meteor, satellite, aircraft, artifact or unknown) and decision (**Keep**, **Omit** or a **Reset to default** entry) are chosen separately in the two drop-down lists below the candidates; **Apply to selected** applies them to every selected candidate. While the list or the image has focus, K keeps, O omits and U resets the selected candidates, and Enter or **Next to review** moves to the next candidate without a choice. With a filter chosen, **Everything in filter…** keeps or omits every candidate the filter shows, after asking for confirmation. **Undo last change** steps back through your changes, bulk ones included; **Clear choices for selected items** removes the selected candidates' entries in **Your choice**.
- **Saving.** **Save draft** lets you continue later; closing the window with unsaved choices asks first. **Apply choices and render** applies the batch of decisions and saves a new photo, leaving the previous result intact.
- **Composite list.** **Export composite list…** saves the meteors added back in the rendered with-trails version as CSV and JSON; see “Meteor showers”.
- **Trail layer.** With **Export trail layer after adding back** ticked and at least one trail kept, `trails_layer_rgba16.tif` is saved in the with-trails folder: a transparent layer containing only the kept trails. Placed in Normal blend mode over the base `final_srgb16.tif`, it reproduces the with-trails photograph exactly. Its values are encoded sRGB display codes, not scene-linear data.

In new reviews, identity drives the default: relabelling a meteor as something else drops its default keep, and relabelling a reliably measured trail as a meteor adds it back. An explicit keep or omit always wins. Unmeasured candidates are never added back. Unknown does not mean satellite, and it does not mean meteor. Older jobs and review sessions keep the old rule: only candidates explicitly marked **keep** are added back. When a review keeps nothing, no new folder is written and Aester reports that the photograph matches the base version. Omit means no additional replay; it does not guarantee removal of a trail already present in the base stack. Keep also cannot guarantee recovery of a faint tail that was never measured. Exposure comparisons help judge continuity; their greyscale view is not the final colour.

The reviewed-photo workflow currently supports colour-calibrated camera RAW. For uncalibrated FITS, candidate records remain available but this calibrated photo-replay path is not supported.

<a name="milkyway"></a>

## Emphasise the Milky Way

In the region editor, choose **Draw Milky Way ellipse** and drag an ellipse around the visible Milky Way; a pink-purple double line marks its extent. Drag its centre and yellow handles to move, resize and rotate it. The square yellow handle and the **Milky Way feathering** slider control feathering. The ellipse may extend beyond the image; zoom out to reach outside handles. Shift-drag to draw again.

Choose **Use these regions** to make Milky Way emphasis available in ③ Optional effects, with a default strength of 60%. No additional confirmation is needed. Leave winter scenes without a visible Milky Way unmarked; **Clear Milky Way** disables the region. A drawing records an editing choice, not an automated identification.

<a name="settings"></a>

## Processing backends

During Setup, select **Aester modified build** to preserve valid zeros, negative values and extra highlight range during stacking. Its runtime, corresponding source and licenses are included for offline use. Sources stay compressed in the component directory, outside your photo output. Run Setup again to add it later.

In **Settings → Processing backends**, choose **Official Siril**, **Siril · Aester modified build**, or a custom compatible build. The installed modified build is found automatically; select it to use it for new jobs. Installing it does not change an existing backend choice. This panel also opens its source and licenses and can import an existing verified build folder. Official Siril remains available.

Unsupported compressed RAW can be imported through the optional **Adobe DNG Converter**, installed separately from the official Adobe link in Settings. Aester finds common installation locations or lets you select the executable. It creates lossless mosaic DNG working copies in the cache and preserves original RAW files. Not all manufacturer-specific metadata is guaranteed to survive; keep your originals. The converter is not included in the Aester installer.

The optional Aester valid-pixel stacking engine is a customised Siril build that distinguishes genuinely zero-valued pixels from edges with no exposure coverage. It currently requires a build verified by this project; an arbitrary executable with the same name is not interchangeable.

For model layout, colour order and other advanced options, use the configuration already verified for that model. Do not infer these from the appearance of a result. **Settings → Components & prompts** shows installation status and offers repair. The nox ONNX model is the project's verified conversion of the author's public weights; another same-named file is not automatically compatible.

<a name="storage"></a>

## Saving, stopping and resuming

Finished photographs default to an Aester folder under your system Pictures folder. To change the destination, use **Choose** beside **Save photographs to** at the bottom of the right column, or **File → Change output folder…**. Each job and reviewed version has its own directory. Its base content is exactly three images: JPEG, 16-bit finished TIFF and float32 linear master TIFF; any extra layers you asked for (starless image, star layers or trail layer) are saved beside them. Logs, recipe records and integrity records stay in the work cache rather than cluttering the photographs folder.

**JPEG quality** at the bottom of the right column (60–100, default 95) sets the quality of the JPEGs a new job delivers. It is saved with the job when stacking starts, and trails versions saved later keep it. Drag the slider, use the arrow keys, or type a value and press Enter; double-click the slider to return to 95. Below it, a rough size per JPEG is shown; with trails on, the trails version saves another JPEG of about the same size. When you open an existing job, the slider shows the quality saved with it; moving the slider then affects new jobs only. 16-bit TIFFs and masters are unaffected; JPEGs saved from **Star reduction** (**Render reduced-star photograph**) and **Local Milky Way enhancement** use the same quality.

Large intermediate files are kept separately in the local work cache, normally under your user account's LocalAppData / Aester / Cache. **Settings → Photographs and cache** changes the cache location for new jobs; a local SSD with sufficient space is recommended. Existing jobs are not moved automatically.

The delivered images can be viewed and edited independently. New linear masters store their colour information and saturation-protection region inside the TIFF, so copying the master alone is enough to render it again in Aester. A master does not contain all RAW exposures or the complete processing history, and is not a backup of the originals. Keep older-format delivery folders together.

Processing runs in a separate process. Closing the window does not cancel it; use **Stop** at the bottom of the right column when you want processing to stop. Progress appears in the readout strip at the bottom of the window; while a job runs, it shows the current stage and the time elapsed, with no estimate of the time remaining. After you stop a job or it fails, logs and completed stages are retained; **Resume job** in the readout strip at the bottom of the window, or **File → Resume unfinished job…**, continues eligible jobs from their checkpoint. If the original files or configuration have changed, start a new job instead.

**File → Save editing project** records the photo list, regions, colour and effects. Open the editing project to continue adjusting them. A project refers to the originals and local preview cache; it does not copy every photo. Keep these referenced files when moving to another computer.

**Settings → Photographs and cache** lists the jobs in your cache folders with each job's size, the space it can free and its status. Select one job and click **Open this job**, or double-click it, to view its photographs or continue its trail review in the main window. **File → Recent jobs** lists the last 10 jobs started or opened in the Aester window. With default settings each 45 MP RAW file takes about 0.9 GB of cache: a 48-frame job usually leaves tens of GB and 150 frames about 150 GB, most of it per-frame intermediate files. Cache scales roughly with pixel count (estimate): about 0.5 GB per photo at 24 MP and 1.3 GB at 61 MP. Once photos are added, a line above **Start stacking** reads like “Start needs ~X GB free · cache drive has Y GB”. It uses the same calculation as the space check at start, from your files' actual pixel counts and with a safety margin; hover over it for the breakdown. When the space is too small, the line turns red and says how many photos fit in one segment. If the space is still too small when stacking starts, the job does not start. Select finished jobs, click **Clean intermediate files of selected jobs…** and confirm: only per-frame files that can be rebuilt from the originals are deleted, namely converted and registered frames, sequence files, copies of the originals, stacking trial and input frames, and FITS debayering frames. Photographs, originals, masters, processing records, logs and unknown files are all kept, and nothing else is deleted automatically.

Before cleaning, Aester verifies the photographs folders. The originals may already have been moved elsewhere, but when an original is no longer at its recorded location (or its size changed), its copy in the cache is kept, so the last copy of a photo is never deleted. A job that is running, whose trails are being reviewed or that another window is using cannot be cleaned; try again later. If another program holds a file open, that file is kept and the list shows the job as partly cleaned; close that program and clean again.

After cleaning, the photographs can still be viewed, and you can save a new preview and redo star tidying. The job's trail candidates can no longer be reviewed, and the job cannot be resumed or have its trail compositing redone; start a new job from the originals when needed. Older jobs that keep their photographs inside the cache keep their original storage layout and are not cleaned here.

## Uninstalling and freeing space

Exit Aester, disconnect any AI assistant still using it and wait for background processing to finish before uninstalling through Windows Installed apps. If the installation is in use, the uninstaller asks you to close those processes and retry.

By default, uninstall removes the program, components exclusive to this installation and registered download files. Originals and finished photographs are preserved. Files you placed in the installation folder, your own models, and older components whose ownership cannot be verified are retained.

You can additionally select **Remove large cache files from completed jobs** and **Remove Aester preferences**. Both options start off. Removing preferences also forgets the **Meteor shower** setting and the **Recent jobs** list. Cache cleanup verifies the independent photographs first and removes only recorded intermediate products. A cleaned job cannot resume or generate another reviewed version; process the originals again when needed. Logs, unknown files and RAW files copied into the cache are preserved, so cleanup may leave a partly populated folder. The two cleanups remove different files: a job already cleaned under **Settings → Photographs and cache** can still have its remaining masters and other image products removed during uninstall.

If cleanup is interrupted after a job is marked as cleaned, a retry may conservatively retain that job's remaining files. It does not promise to erase every leftover cache file. Unfinished installation preparation and other unowned files may also remain and are recorded in the uninstall report.

<a name="retouch"></a>

## Natural-language retouching

Choose **Let an Agent retouch** at the end of the stage bar (06 Refine). Under **What you want**, describe what to change and preserve, check the prompt that will be copied for the Agent, and copy it to an Agent that can inspect images and use local MCP tools. You can open this before stacking. When a finished photo is available, its TIFF is included. When trails were added back, the with-trails TIFF is selected automatically, and a hint line below it says which version is current. You can still choose another file; a file you choose yourself is kept.

Aester only prepares the prompt; it does not start an Agent. Your Agent reads the finished photo and retouches it through Aester's main MCP service. Retouched versions are saved separately in the photographs folder, and the original finished photo is kept.

Under **Reference image (optional)** you can add an image whose look you like and say where it came from: **Someone else’s work**, **My own work** or **AI-generated image**. Put it inside the material scope the assistant can read or inside the photographs folder, otherwise the Agent may not be able to open it; the dialog warns you. The Agent uses it only for look and goals, such as warmth, tonal priorities, Milky Way layering and structure, and never copies its pixels. Without a reference, the Agent asks you once whether you have one.

1. Under **Settings → Components & prompts**, tick GIMP and the retouching skill you want prepared, or choose an existing GIMP console program, then choose the highlighted **Generate the assistant prompt for the selected components**. It prepares what is ticked first (for GIMP, an existing program on this computer is used; otherwise the official GIMP installer is downloaded from the official site or the fastest listed mirror and checked against its SHA-256), then writes and copies the prompt. With nothing ticked, it only writes and copies the prompt. The hint under the button lists what will be prepared. GIMP performs the local retouching operations. Its official installer manages it separately, and uninstalling Aester preserves GIMP. Basic stacking does not need it.
2. Preparing the **Aester retouching skill** is recommended. Ask the Agent to read the skill entry listed in the prompt, or install it using your client's instructions. The skill guides image inspection, detail checks and candidate comparisons. You can still copy prompts and use MCP without installing the skill. Aester does not automatically write to your client's skill directory.
3. Copy the configuration from **Connect an AI assistant**, add it to your client and verify the connection. The retouching prompt includes the local skill entry, component information, selected photo and your request. Copying it does not start an Agent or establish a client connection.
4. Ask the Agent to inspect the photo and available operations, make small adjustments, and show both the full frame and matching 100% details. Give concrete feedback, such as “Show more Milky Way structure, preserve airglow, brighten the landscape slightly and keep bright stars from growing.” Select a candidate, then save new JPEG, TIFF and editable XCF files while retaining the original.

This entry uses a 16-bit RGB finished TIFF with an explicit sRGB colour profile. RAW files, linear masters and images with unclear colour encoding need an explicit display conversion first. An ordinary TIFF does not automatically contain separate star, Milky Way or bright-trail layers; those operations need suitable real layer data. Star shape, noise and complex boundaries still have limitations. Successful replay does not establish that a photo meets your aesthetic goals.

The local executor processes the photographs. Whether an external Agent client sends images to a model service depends on that client, your request and your authorisation. The skill package contains no practice photographs.

<a name="mcp"></a>

## Let an AI assistant help through MCP

GUI and MCP use the same processing core and parameters; you do not need to learn a second photography workflow. Open **Settings → Connect an AI assistant**, choose the input folders the assistant is allowed to read, and copy the configuration or setup prompt.

**Test local service** actually starts the service and checks tool calls. A successful test means that local Aester responds; it does not mean your Agent client is connected. Add the configuration to the client and ask the assistant to verify the connection. Aester needs no model API key and does not modify your client configuration automatically.

Try these prompts:

> Stack the night-sky photos I selected. Use only the files I list and use the first as the colour reference. Deliver a linear master, a 16-bit finished TIFF and a JPEG. Leave trail replay and star reduction off initially. Before starting, tell me whether sky/foreground regions are still needed. When finished, show me the photographs and their location.

> These photos show only sky, with no foreground. Reduce stars using the recommended strength; tell me first if a model is missing. Make the colour slightly warmer and increase exposure by 0.3 EV. Do not enhance an unconfirmed Milky Way.

> Mark sky and foreground on the calibrated first-frame preview and show me the regions before starting. Draw a movable, rotatable, feathered ellipse over the visible Milky Way and preview it. The drawn region can then be enhanced. Leave it undrawn when absent or uncertain.

> Scan this sequence for bright trails. First show me the base photo and candidates, listing unknown identities separately. Inspect each candidate's exposure and neighbouring exposures. Save my batch of keep/omit decisions, then generate a new version without overwriting the previous photo.

For integrators, the main entry points are `get_help`, `inspect_sources`, `prepare_reference`, `submit_workflow` and `workflow_status`. Submission accepts an explicit `input_files` list. `trails_enabled` and `stars_enabled` both default to false.

Prefer `draw_regions` to draw horizon points, foreground strokes and a Milky Way ellipse and return an overlay. Lower-level `create_regions`, `update_regions` and `get_regions` remain available. `preview_regions` generates the same overlay used by the GUI, with orange for the foreground and a pink-purple double line for the Milky Way ellipse. It returns paths by default; only an explicit `include_image=true` returns a small image to the current MCP client, subject to the user's authorisation to share images. A preview alone does not enable the Milky Way. Explicit drawing enables the region and records the actual editor source, without another human-confirmation switch. Pass the returned region-project path to `submit_workflow(regions_path=...)`.

Trail review uses `open_trail_review`, `trail_candidate_context`, `save_trail_review` and `apply_trail_review`. Save and apply include the draft revision just read, preventing another window's changes from being overwritten. MCP records the source as an Agent, which the interface marks with a pale-gold dot and never with the 人定 seal. Agent drawings never masquerade as human judgment. `import_photos` uses the host-configured Adobe converter for unsupported RAW; its explicit DNG list can continue through preview, regions and stacking, without allowing arbitrary cache inputs.

`prepare_reference` returns `image_rotation`, the total clockwise angle relative to unrotated decoded pixels. To correct it, call `rotate_reference(reference_dir, image_rotation=0/90/180/270)` and redraw regions on the new reference. Submit the same angle, or let `regions_path` bind it automatically. Originals and old regions remain unchanged. A standalone legacy PNG mask retains native orientation unless `image_rotation` is supplied explicitly.
`color_adjustments` contains `warmth`, `tint` (−100 to 100), and `exposure_ev` (−4 to 4). Zero is the default. Protocol keys and numeric settings are identical in both interface languages.
`jpeg_quality` (an integer from 60 to 100) sets the quality of every JPEG the job delivers, with 4:4:4 chroma. It is frozen in the submission and later trails versions keep it; when omitted, each JPEG keeps its previous encoding. The interface always sends the current **JPEG quality** slider value.

The host limits input, photographs and cache directories and configures the engine and model. Generated configuration uses `STARSTACK_INPUT_ROOT`, `STARSTACK_OUTPUT_ROOT` and `STARSTACK_CACHE_ROOT`; tools cannot enlarge these roots. `AESTER_LANGUAGE` sets the independent service's help language. Older configurations without a cache root retain the old storage layout. Keep the job location returned by every submission; reuse its submission identifier when retrying to avoid duplicate large jobs.

`inspect_workflow_cache(workflow_dir)` reports, read-only, which intermediate files of a finished job could be freed, what is kept and why a job cannot be cleaned. Inspection does not require the originals to be in place. Only the user can delete them, after confirming under **Settings → Photographs and cache**; MCP offers no deletion tool. A cleaned job carries `cache_cleaned` in `workflow_status`.

Advanced tools remain available, but beginners should not have to learn internal intermediate-file names. A local preview is not automatically uploaded to an external model.

<a name="features"></a><a name="raw"></a><a name="background"></a><a name="layers"></a><a name="details"></a><a name="limits"></a><a name="about"></a>

## Current limits

This is a preview version. Principal real-world validation covers Nikon Z8 wide-field scenes and one Seestar deep-sky sequence. Decoding another camera model does not establish photographic quality for that model.

Complex foreground joins, general meteor identification, production-quality starless works and fully automatic Milky Way localisation have not all passed acceptance. A Windows installation candidate is available, but installation and real material still need testing on other computers.

If a job fails, the readout strip at the bottom of the window shows the full message; choose **Details** to inspect the processing record. Include the type of scene, selected options, job directory and location of the visible problem when reporting it.
'''

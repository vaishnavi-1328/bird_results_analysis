Optimize the attached bird-detection code for **maximum accuracy and substantially faster processing**.

### Hard requirements

- Birds can be extremely distant (or close)and only **10- 11 pixels**, so **do not remove adaptive thresholding or reduce resolution** in a way that loses them.
- **Every real bird should be detected**, while **false detections must also be minimized**.
- Do not gain speed simply by skipping many frames or weakening detection.
- Target: process a **10+ hour / ~10 GB video in under 12 hours**.

### Focus specifically on these problems in the current code

**1. Full-frame Farneback optical flow is wasteful.**

The code calculates dense optical flow over the entire sky ROI even though flow is ultimately inspected only around detected candidates.

Change this to **candidate/local-region motion analysis**. Compute motion only around candidate bounding boxes with sufficient padding. Verify that this preserves detection of tiny birds.

**2. The 1×, coarse 1×, 1.5× and 2× adaptive-threshold passes are expensive.**

Do NOT blindly remove them.

First determine which passes actually produce unique valid bird detections. Optimize them by:
- reusing grayscale/blurred data,
- avoiding repeated full-frame conversions/resizes,
- avoiding redundant calculations,
- and, where possible without reducing recall, performing expensive upscale analysis only on relevant regions.

Adaptive thresholding must remain sensitive enough for 1–3 px birds.

**3. False candidates must be rejected early.**

Do not solve recall by generating enormous numbers of candidates and sending all of them through expensive processing.

Reorder/optimize filters so cheap high-confidence rejection happens first:

`adaptive candidates → geometry/area → contrast/darkness → sky validity → temporal/motion evidence → expensive cloud/motion analysis → tracker`

Determine the best ordering from the existing code rather than following this ordering blindly.

**4. Remove repeated image processing.**

Look for repeated:
- BGR→GRAY conversions
- BGR→HSV conversions
- Gaussian blurs
- resizing
- ROI copying
- cloud calculations
- contour calculations

Compute reusable frame-level representations once and reuse them.

**5. Optimize ROI calculation.**

The horizon currently does not need expensive recalculation every few frames if the camera/scene has not changed. Make ROI updates less frequent or change-triggered without excluding valid sky.

**6. Optimize tracking.**

Inspect the nested candidate-to-track matching loops. Improve them if they become expensive with many candidates, while preventing:
- duplicate bird counts,
- merging different birds,
- losing temporarily disappearing birds.

**7. Remove unnecessary streaming/display overhead during batch processing.**

Do not copy frames, draw boxes/text, or yield preview frames unless preview is actually requested.

**8. Profile before and after.**

Add timing instrumentation around the major stages:

`decode | adaptive threshold | upscale passes | contours/filters | MOG2 | optical flow | tracking | display`

Report milliseconds/frame and percentage of runtime.

Then optimize the actual bottlenecks.

### After single-thread optimization

If additional speed is required, implement **parallel video-chunk processing with overlap**, because tracking/MOG2/previous-frame state prevents arbitrary frame-level multiprocessing.

Ensure overlapping chunks do not double-count birds crossing boundaries.

Also investigate hardware-accelerated video decoding and OpenCV/GPU acceleration if available.

### Most important rule

Do **not** make an optimization unless it preserves both:

- **Recall:** real birds, especially 1–3 px distant birds, are not lost.
- **Precision:** clouds, noise, compression artifacts, insects, etc. do not become birds.

Use representative frames/video segments to compare the original and optimized implementations.

### Deliverable

Modify the actual code, not just explain recommendations.

Give me:

1. The main bottlenecks you found.
2. What you changed and why.
3. The optimized code.
4. Before/after processing speed.
5. Any change that could potentially affect detection accuracy.

Prioritize first:
**localizing optical flow → eliminating redundant image operations → optimizing multi-scale adaptive thresholding → early false-candidate rejection → ROI optimization → tracking optimization → parallelization.**
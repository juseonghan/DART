# DART project page

Static site for **DART: Depth-as-Target Pretraining for Surgical Vision Foundation Models** (BMVC 2026).
No build step: plain HTML/CSS/JS plus image tiles in `static/img/`.

## 1. Fill in your links
Open `index.html` and edit the block near the top:

```js
window.DART_LINKS = {
  paper:   "",   // BMVC proceedings PDF
  arxiv:   "",   // https://arxiv.org/abs/XXXX.XXXXX
  code:    "",   // https://github.com/<you>/DART
  weights: "",   // https://huggingface.co/<you>/DART-ViT-B16
};
```
An empty value shows a greyed-out button with a "soon" tag, so you can publish before everything is public.

Also check the "Extract features" snippet: the checkpoint filename (`dart_vitb16.pth`)
and key (`teacher_backbone`, which is what `train.py` saves) should match the weights you release.

## 2. Publish with GitHub Pages
**Option A, in the code repo (recommended):** copy `index.html`, `.nojekyll` and `static/` into a
`docs/` folder of the DART repo, push, then go to *Settings → Pages → Build and deployment →
Deploy from a branch*, pick `main` and `/docs`. The page appears at `https://<you>.github.io/DART/`.

**Option B, a separate repo:** push these files to the root of a new repo, then set Pages to `main` and `/ (root)`.

## Preview locally
```bash
python -m http.server 8000   # then open http://localhost:8000
```

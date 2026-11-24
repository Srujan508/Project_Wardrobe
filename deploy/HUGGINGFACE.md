# Deploying to a Hugging Face Space

The app runs as a Docker Space on the free CPU tier. Nothing in this repo pushes automatically.

## 1. Build the Space folder

```powershell
venv\Scripts\python -m scripts.prepare_space --model B
```

This writes `dist/space/` with the Dockerfile, the code, model `B` and its calibrator (about 10 MB), plus a Space card.

## 2. Create the Space and push

1. On huggingface.co, go to **New Space**. Pick **Docker** as the SDK, use a blank template, and make it **Private** to start.
2. Push the folder:
   ```powershell
   cd dist\space
   git init -b main
   git add .
   git commit -m "AI Wardrobe Stylist"
   git remote add origin https://huggingface.co/spaces/<you>/<space-name>
   git push -u origin main
   ```

## 3. Set secrets before sharing the link

Go to **Settings → Variables and secrets** and add:

| Name | Value | Why |
|---|---|---|
| `WARDROBE_BASIC_AUTH` | `user:long-random-password` | The app has no accounts, so without this anyone with the link can see and delete the wardrobe. |
| `WARDROBE_ALLOWED_HOSTS` | `<you>-<space-name>.hf.space` | The host allow-list (DNS-rebinding protection). |

## Notes

- **Storage:** free Spaces don't keep files. Uploads and the database reset when the Space restarts.
- **Style study:** it needs the Polyvore test images, which are not shipped (the dataset is gated), so the study tab is hidden in the Space.
- **Speed:** on CPU, the first upload takes 1–2 minutes to load the models. Later uploads take a few seconds each.

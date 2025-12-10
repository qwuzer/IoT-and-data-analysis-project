# Quick Start: Real-Time Pipeline

## Quick Setup

1. **Ensure you have a trained LSTM model checkpoint**:
   ```bash
   # Should exist at:
   speed_estimation/LSTM/models/checkpoints/best_model.pt
   ```

2. **Run the pipeline**:
   ```bash
   python speed_estimation/realtime_pipeline.py \
       --lstm_checkpoint speed_estimation/LSTM/models/checkpoints/best_model.pt \
       --webpage_url https://tw.live/cam/?id=BOT243
   ```

3. **Press 'q' to quit** when done

## Minimal Example

```bash
python speed_estimation/realtime_pipeline.py \
    --lstm_checkpoint <path_to_model.pt> \
    --webpage_url <stream_webpage_url>
```

## What You'll See

- Video window showing live stream
- Vehicles detected with colored boxes
- Labels showing: Vehicle ID, type, speed, and STOP/GO prediction
- Traffic light status indicator
- Vehicle movement trails

## Troubleshooting

**No predictions showing?**
- Wait ~12 frames for each vehicle (sequence needs to build up)
- Check console for errors

**Stream won't open?**
- Try using `--stream_url` with direct m3u8 URL
- Check network connectivity

**Low FPS?**
- Use `--imgsz 640` for faster processing
- Ensure GPU is available if you have one

See `REALTIME_PIPELINE.md` for full documentation.


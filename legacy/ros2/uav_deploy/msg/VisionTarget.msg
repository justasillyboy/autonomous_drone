# One detected target — populated by perception_node after all filters pass.
# Units: pixels in the frame coordinate system (origin = top-left).
uint8   cls_id          # 0=circle, 1=semicircle, 2=H_landing (indices from dataset.py CLASS_NAMES)
string  color           # "red" | "yellow" | "blue" | "" (empty when colour detection failed / N/A)
float32 cx              # centre X in pixels (frame coords)
float32 cy              # centre Y in pixels
float32 radius          # enclosing-circle radius in pixels
float32 fill_ratio      # mask pixels / solid circle area (already ≥ FILL_RATIO_MIN_CIRCLE)
float32 z_est_raw       # vision-only altitude estimate in metres (f_px·D_real / (2·radius)); fallback only
float32 conf            # YOLO detection confidence
bool    edge_clipped    # True → circle touches frame border; radius is untrustworthy

import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path

df = pd.read_parquet("my_clothes_desc.parquet")

out_dir = Path("images")
out_dir.mkdir(exist_ok=True)

for idx, row in df.iterrows():
    img_dict = row["image"]
    img_bytes = img_dict["bytes"]           
    
    img = Image.open(BytesIO(img_bytes))
    
    img.save(out_dir / f"img_{idx}.jpg")

print(f"Salvas {len(df)} imagens em {out_dir.resolve()}")

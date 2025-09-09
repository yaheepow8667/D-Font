import torch
from model.dp_model import DPFontModel
from utils import save_batch

def run_sample(ckpt_path, content_list, stroke_list, style_list, out_dir='samples'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DPFontModel(device=device).to(device)
    data = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(data['model'])
    content = torch.tensor(content_list, dtype=torch.long, device=device)
    stroke = torch.tensor(stroke_list, dtype=torch.long, device=device)
    style = torch.tensor(style_list, dtype=torch.long, device=device)
    with torch.no_grad():
        x = model.sample(content, stroke, style, T=100, guidance_scale=3.5, device=device)
        save_batch(x, out_dir, prefix='sample')

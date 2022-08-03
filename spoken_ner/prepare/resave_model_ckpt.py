import fire
import os
import torch
import fairseq.models.wav2vec.wav2vec2 as w2v


def load_ckpt(ckpt_pth):
    with open(ckpt_pth, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    return state


def get_diff(state_pt, state_ft, pfx):
    keys_ft = list(state_ft["model"].keys())
    keys_pt = list(state_pt["model"].keys())
    in_keys, out_keys, diff = [], [], 0
    for key in keys_pt:
        corresponding_ft_key = pfx + key
        if corresponding_ft_key in keys_ft:
            in_keys.append(key)
            diff += torch.norm(
                state_pt["model"][key] - state_ft["model"][corresponding_ft_key]
            )
        else:
            out_keys.append(key)
    diff = diff / len(in_keys)
    return diff, in_keys


def resave_ckpt(ft_model_ckpt_dir, pt_model_ckpt):
    state_ft = load_ckpt(
        os.path.join(ft_model_ckpt_dir, "checkpoints", "checkpoint_best.pt")
    )
    state_pt = load_ckpt(pt_model_ckpt)
    pfx = "w2v_encoder.w2v_model."
    diff_prev, in_keys = get_diff(state_pt, state_ft, pfx)

    for key in in_keys:
        state_pt["model"][key] = state_ft["model"][pfx + key]
    diff_new, _ = get_diff(state_pt, state_ft, pfx)

    print(
        "[Sanity check, difference in parameter norms] Before: %.2f,After: %.2f"
        % (diff_prev, diff_new)
    )
    save_dir = os.path.join(ft_model_ckpt_dir[:-1] + "_no_proj", "checkpoints")
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state_pt, os.path.join(save_dir, "checkpoint_best.pt"))

    print(f"{ft_model_ckpt_dir} model saved without projection head")


if __name__ == "__main__":
    fire.Fire(resave_ckpt)
